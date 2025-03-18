import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Optional, Tuple, Union, Any
import functools


class ResNetBlock(nn.Module):
    """A ResNet block with batch normalization."""
    features: int
    strides: Tuple[int, int] = (1, 1)
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        
        # First conv layer
        y = nn.Conv(features=self.features, kernel_size=(3, 3), strides=self.strides, padding=((1, 1), (1, 1)))(x)
        y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)
        
        # Second conv layer
        y = nn.Conv(features=self.features, kernel_size=(3, 3), padding=((1, 1), (1, 1)))(y)
        y = nn.BatchNorm(use_running_average=not training)(y)
        
        # Handle different dimensions for residual connection
        if residual.shape != y.shape:
            residual = nn.Conv(features=self.features, kernel_size=(1, 1), strides=self.strides)(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        # Add residual connection and apply ReLU
        return nn.relu(y + residual)


class ResNet5(nn.Module):
    """A 5-layer ResNet with batch normalization."""
    features: Sequence[int] = (32, 32)
    avg_pooling: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):        
        # ResNet blocks
        for i in range(len(self.features)):
            x = ResNetBlock(features=self.features[i], strides=(2, 2) if i > 0 else (1, 1))(x, training=training)
        
        # Average pooling
        if self.avg_pooling:
            x = nn.avg_pool(x, window_shape=x.shape[1:3], strides=(1, 1))
        x = x.reshape(x.shape[0], -1)  # Flatten
        
        return x

class SimpleScan(nn.Module):
    hidden_size: int
    @nn.compact
    def __call__(self, c, xs):
        LSTM = nn.scan(nn.OptimizedLSTMCell,
                   variable_broadcast="params",
                   split_rngs={"params": False},
                   in_axes=1,
                   out_axes=1)
        return LSTM(self.hidden_size)(c, xs)

class CharacterNet(nn.Module):
    """
    A network that processes visual trajectories through a ResNet backbone,
    followed by an LSTM for temporal processing, and a final linear layer.
    
    Args:
        output_size: Size of the output vector (2 or 8)
        lstm_hidden_size: Size of the LSTM hidden state
    """
    output_size: int
    lstm_hidden_size: int = 64
    
    @nn.compact
    def __call__(self, observations, actions, training: bool = True):
        # inputs shape: (batch, trajectory_length, h, w, 3)
        batch_size, trajectory_length = observations.shape[0], observations.shape[1]
        
        # Reshape to process all frames through ResNet
        reshaped_observations = observations.reshape(-1, *observations.shape[2:])  # (batch * trajectory_length, h, w, 3)
        reshaped_actions = actions.reshape(-1, *actions.shape[2:])  # (batch * trajectory_length, 1)
        
        # Convert actions to one-hot encoding (assuming 6 possible actions)
        one_hot_actions = jax.nn.one_hot(reshaped_actions.squeeze(-1), num_classes=6)  # (batch * trajectory_length, 6)
        # Get spatial dimensions from observations
        h, w = reshaped_observations.shape[1:3]
        
        # Tile the one-hot actions to match spatial dimensions
        tiled_actions = jnp.tile(one_hot_actions[:, None, None, :], (1, h, w, 1))  # (batch * trajectory_length, h, w, 6)
        
        # Concatenate observations and actions along the last axis
        combined_input = jnp.concatenate([reshaped_observations, tiled_actions], axis=-1)  # (batch * trajectory_length, h, w, 3+6)
        
        # Process through ResNet
        resnet = ResNet5(avg_pooling=True)
        visual_features = resnet(combined_input, training=training)
        
        # Reshape back to separate batch and time dimensions
        visual_features = visual_features.reshape(batch_size, trajectory_length, -1)
        
        # Process sequence through LSTM
        lstm = SimpleScan(hidden_size=self.lstm_hidden_size)
        def init_carry(batch_size, hidden_size):
            return (jnp.zeros((batch_size, hidden_size)), jnp.zeros((batch_size, hidden_size)))
        init_carry = init_carry(batch_size, self.lstm_hidden_size)
        final_carry, final_output = lstm(init_carry, visual_features)

        # Take the final LSTM output at the last time step
        final_output = final_output[:, -1, :]
        
        # Final linear layer
        character_embeddings = nn.Dense(features=self.output_size)(final_output)
        
        return character_embeddings.sum(axis=0)  # sum up character embeddings to get a single vector

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell that preserves spatial dimensions."""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    
    @nn.compact
    def __call__(self, carry, inputs):
        c, h = carry
        batch_size, height, width, _ = inputs.shape
        
        # Concatenate the input and hidden state along the channel dimension
        concat_input = jnp.concatenate([inputs, h], axis=-1)
        
        # Compute the gates using convolutions
        gates = nn.Conv(
            features=4 * self.features,
            kernel_size=self.kernel_size,
            padding='SAME'
        )(concat_input)
        
        # Split the gates
        i, f, o, g = jnp.split(gates, 4, axis=-1)
        
        # Apply nonlinearities
        i = jax.nn.sigmoid(i)  # input gate
        f = jax.nn.sigmoid(f)  # forget gate
        o = jax.nn.sigmoid(o)  # output gate
        g = jnp.tanh(g)        # cell update
        
        # Update cell state
        new_c = f * c + i * g
        
        # Compute hidden state
        new_h = o * jnp.tanh(new_c)
        
        return (new_c, new_h), new_h

class ConvLSTMScan(nn.Module):
    """Convolutional LSTM that processes sequences while preserving spatial dimensions."""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    
    @nn.compact
    def __call__(self, carry, xs):
        # Use nn.scan to apply the ConvLSTMCell across the time dimension
        ConvLSTM = nn.scan(
            ConvLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )
        return ConvLSTM(self.features, self.kernel_size)(carry, xs)

class MentalNet(nn.Module):
    """
    A network that processes visual trajectories through a ResNet backbone,
    followed by a convolutional LSTM for temporal processing that preserves spatial dimensions.
    
    Args:
        output_channels: Number of output channels for the mental state embedding
    """
    output_channels: int = 32
    
    @nn.compact
    def __call__(self, observations, actions, training: bool = True):
        # inputs shape: (batch, trajectory_length, h, w, 3)
        batch_size, trajectory_length = observations.shape[0], observations.shape[1]
        
        # Reshape to process all frames through ResNet
        reshaped_observations = observations.reshape(-1, *observations.shape[2:])  # (batch * trajectory_length, h, w, 3)
        reshaped_actions = actions.reshape(-1, *actions.shape[2:])  # (batch * trajectory_length, 1)
        
        # Convert actions to one-hot encoding (assuming 6 possible actions)
        one_hot_actions = jax.nn.one_hot(reshaped_actions.squeeze(-1), num_classes=6)  # (batch * trajectory_length, 6)
        
        # Get spatial dimensions from observations
        h, w = reshaped_observations.shape[1:3]
        
        # Tile the one-hot actions to match spatial dimensions
        tiled_actions = jnp.tile(one_hot_actions[:, None, None, :], (1, h, w, 1))  # (batch * trajectory_length, h, w, 6)
        
        # Concatenate observations and actions along the last axis
        combined_input = jnp.concatenate([reshaped_observations, tiled_actions], axis=-1)  # (batch * trajectory_length, h, w, 3+6)
        
        # Process through ResNet
        resnet = ResNet5(avg_pooling=False)
        visual_features = resnet(combined_input, training=training)
        
        # Reshape back to separate batch and time dimensions
        visual_features = visual_features.reshape(batch_size, trajectory_length, h, w, -1)
        
        # Process sequence through convolutional LSTM
        conv_lstm = ConvLSTMScan(features=self.output_channels)
        
        # Initialize carry state with zeros
        def init_carry(batch_size, height, width, features):
            return (
                jnp.zeros((batch_size, height, width, features)),  # cell state
                jnp.zeros((batch_size, height, width, features))   # hidden state
            )
        
        init_carry = init_carry(batch_size, h, w, self.output_channels)
        
        # Check if trajectory is empty (initial state)
        # We'll use a condition to either return zeros or process through LSTM
        is_empty = trajectory_length == 0
        
        # If trajectory is not empty, process through LSTM
        final_carry, lstm_outputs = conv_lstm(init_carry, visual_features)
        
        # Take the final LSTM output at the last time step
        final_output = lstm_outputs[:, -1]  # shape: (batch, h, w, output_channels)
        
        # Apply a 1-layer convnet with output_channels
        mental_embedding = nn.Conv(
            features=self.output_channels,
            kernel_size=(3, 3),
            padding='SAME'
        )(final_output)  # shape: (batch, h, w, output_channels)
        
        # Create a condition to handle empty trajectories
        # If trajectory is empty, return zeros
        zero_embedding = jnp.zeros((batch_size, h, w, self.output_channels))
        mental_embedding = jnp.where(is_empty, zero_embedding, mental_embedding)
        
        return mental_embedding
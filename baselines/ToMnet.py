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
        y = nn.LayerNorm()(nn.BatchNorm(use_running_average=not training)(y))
        y = nn.relu(y)
        
        # Second conv layer
        y = nn.Conv(features=self.features, kernel_size=(3, 3), padding=((1, 1), (1, 1)))(y)
        y = nn.LayerNorm()(nn.BatchNorm(use_running_average=not training)(y))
        
        # Handle different dimensions for residual connection
        if residual.shape != y.shape:
            residual = nn.Conv(features=self.features, kernel_size=(1, 1), strides=self.strides)(residual)
            residual = nn.LayerNorm()(nn.BatchNorm(use_running_average=not training)(residual))
        
        # Add residual connection and apply ReLU
        return nn.relu(y + residual)


class ResNet5(nn.Module):
    """A 5-layer ResNet with batch normalization."""
    features: Sequence[int] = (32, 32)
    avg_pooling: bool = True
    flatten: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):        
        # ResNet blocks
        for i in range(len(self.features)):
            x = ResNetBlock(features=self.features[i], strides=(2, 2) if i > 0 else (1, 1))(x, training=training)
        
        # Average pooling
        if self.avg_pooling:
            x = nn.avg_pool(x, window_shape=x.shape[1:3], strides=(1, 1))
        if self.flatten:
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
        one_hot_actions = jax.nn.one_hot(reshaped_actions, num_classes=6)  # (batch * trajectory_length, num_to_predict, 6)
        one_hot_actions = one_hot_actions.reshape(batch_size*trajectory_length, -1)
        # Get spatial dimensions from observations
        h, w = reshaped_observations.shape[1:3]
        
        # Tile the one-hot actions to match spatial dimensions
        tiled_actions = jnp.tile(one_hot_actions[:, None, None, :], (1, h, w, 1))  # (batch * trajectory_length, h, w, 6)
        
        # Concatenate observations and actions along the last axis
        combined_input = jnp.concatenate([reshaped_observations, tiled_actions], axis=-1)  # (batch * trajectory_length, h, w, 3+6*num_to_predict)

        # Process through ResNet
        resnet = ResNet5(avg_pooling=True)
        visual_features = resnet(combined_input, training=training)
        
        # Apply layer normalization
        visual_features = nn.LayerNorm()(visual_features)
        
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
        character_embeddings = nn.LayerNorm()(character_embeddings)
        
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
        
        # Apply layer normalization
        gates = nn.LayerNorm()(gates)
        
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
        batch_size, trajectory_length, h, w = observations.shape[0], observations.shape[1], observations.shape[2], observations.shape[3]
        
        # Reshape to process all frames through ResNet
        reshaped_observations = observations.reshape(-1, *observations.shape[2:])  # (batch * trajectory_length, h, w, 3)
        reshaped_actions = actions.reshape(-1, *actions.shape[2:])  # (batch * trajectory_length, 1)
        
        # Convert actions to one-hot encoding (assuming 6 possible actions)
        one_hot_actions = jax.nn.one_hot(reshaped_actions, num_classes=6)  # (batch * trajectory_length, num_to_predict,6)
        one_hot_actions = one_hot_actions.reshape(batch_size*trajectory_length, -1)
        
        # Get spatial dimensions from observations
        h, w = reshaped_observations.shape[1:3]
        
        # Tile the one-hot actions to match spatial dimensions
        tiled_actions = jnp.tile(one_hot_actions[:, None, None, :], (1, h, w, 1))  # (batch * trajectory_length, h, w, 6)
        
        # Concatenate observations and actions along the last axis
        combined_input = jnp.concatenate([reshaped_observations, tiled_actions], axis=-1)  # (batch * trajectory_length, h, w, 3+6)
        
        # Process through ResNet
        resnet = ResNet5(avg_pooling=False)
        visual_features = resnet(combined_input, training=training)
        
        # Apply layer normalization
        visual_features = nn.LayerNorm()(visual_features)
        
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
        # final_output = lstm_outputs[:, -1]  # shape: (batch, h, w, output_channels)
        final_output = lstm_outputs.reshape(-1, h, w, self.output_channels)

        
        # Apply a 1-layer convnet with output_channels
        mental_embedding = nn.Conv(
            features=self.output_channels,
            kernel_size=(3, 3),
            padding='SAME'
        )(final_output)  # shape: (batch, h, w, output_channels)
        
        # Apply layer normalization
        mental_embedding = nn.LayerNorm()(mental_embedding)
        
        # Create a condition to handle empty trajectories
        # If trajectory is empty, return zeros
        zero_embedding = jnp.zeros((batch_size*trajectory_length, h, w, self.output_channels))
        mental_embedding = jnp.where(is_empty, zero_embedding, mental_embedding)

        mental_embedding = mental_embedding.reshape(batch_size, trajectory_length, h, w, self.output_channels)
        
        return mental_embedding

class ToMNet(nn.Module):
    """
    A network that processes visual trajectories through a ResNet backbone,
    followed by a convolutional LSTM for temporal processing that preserves spatial dimensions.
    
    Args:
    """
    character_net_features: int
    mental_net_features: int
    output_size: int = 6
    num_to_predict: int = 1

    @nn.compact
    def __call__(self, observations, actions, training: bool = True):
        
        # select the first n - 1 observations and actions
        character_observations = observations[:-1]
        character_actions = actions[:-1]
        # (character_net_features,)
        character_embeddings = CharacterNet(output_size=self.character_net_features)(character_observations, character_actions, training=training)

        # take most recent trajectory, and use the first n - 1 observations and actions for mental net
        mental_observations = observations[-1:, :-1]  
        mental_actions = actions[-1:, :-1]  
        # (1, trajectory_length - 1, h, w, mental_net_features)
        mental_embeddings = MentalNet(output_channels=self.mental_net_features)(mental_observations, mental_actions, training=training)

        query_obs = observations[-1:, 1:]  # (1, trajectory_length - 1, h, w, 3)
        # query_action = actions[-1:, 1:]  # (1, trajectory_length - 1, 1)

        # get shared prediction torso
        prediction_torso = PredictionTorso()(query_obs, character_embeddings, mental_embeddings, training=training)

        
        reshaped_pred = prediction_torso.reshape(-1, *prediction_torso.shape[2:])

        # action prediction head
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(reshaped_pred)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=x.shape[1:3], strides=(1, 1))
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = nn.Dense(features=self.mental_net_features)(x)
        x = nn.relu(x)

        action_net = nn.vmap(
            ActionMapper,
            in_axes=(0, None),
            out_axes=0,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )(self.output_size, self.num_to_predict)
            
        agent_action_predictions = action_net(jnp.arange(self.num_to_predict), x)

        return agent_action_predictions  # (num_to_predict, batch*(traj_length - 1), self.output_size)

class ActionMapper(nn.Module):
    """
    A network that maps a set of action predictions to a single action.
    """
    output_size: int
    num_to_predict: int
    @nn.compact
    def __call__(self, agent_idx, joint_action_embed):
        # get the action prediction for the agent
        embed = jnp.zeros(self.num_to_predict)
        embed = embed.at[agent_idx].set(1)
        embed = jnp.repeat(embed, joint_action_embed.shape[0], axis=0)
        embed = embed.reshape(joint_action_embed.shape[0], -1)
        x = jnp.concatenate([joint_action_embed, embed], axis=-1)
        x = nn.Dense(features=self.output_size * 3)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_size * 2)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.output_size)(x)
        action_prediction = nn.softmax(logits)  # (batch*(traj_length - 1), self.output_size)
        return action_prediction

class PredictionTorso(nn.Module):
    """
    A prediction network that processes combined features through a 5-layer ResNet
    with 32 channels, ReLU nonlinearities, and batch normalization.
    
    Args:
        output_size: Size of the output vector
    """
    
    @nn.compact
    def __call__(self, query_obs, character_embeddings, mental_embeddings, training: bool = True):
        
        # Get spatial dimensions from query observation
        batch_size, traj_len, h, w, c = query_obs.shape
        batch_size, traj_len, h, w, mental_net_features = mental_embeddings.shape
        character_net_features = character_embeddings.shape[0]

        
        # Tile character embeddings to match spatial dimensions
        # Reshape from (character_net_features,) to (1, traj_len, h, w, character_net_features)
        tiled_character_embeddings = jnp.tile(
            character_embeddings[None, None, None, None, :], 
            (batch_size, traj_len, h, w, 1)
        )
        
        # Concatenate all embeddings along the final axis
        combined_features = jnp.concatenate([
            query_obs,
            tiled_character_embeddings,
            mental_embeddings
        ], axis=-1)  # (batch_size, traj_len, h, w, 3 + character_net_features + mental_net_features)
        

        # Reshape
        x = combined_features.reshape(batch_size * traj_len, h, w, 3 + character_net_features + mental_net_features)
                
        # 5 ResNet blocks with 32 channels each
        x = ResNet5(avg_pooling=False, flatten=False)(x, training=training)
        
        # Apply layer normalization
        x = nn.LayerNorm()(x)
        
        # Reshape to add back time dimension
        x = x.reshape(batch_size, traj_len, *x.shape[1:])
        
        return x
        
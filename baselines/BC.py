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
    features: Sequence[int] = (32,)
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

class BCNet(nn.Module):
    """
    A network that processes visual trajectories through a ResNet backbone
    
    Args:
        output_channels: Number of output channels for the mental state embedding
    """
    output_size: int = 6
    hidden_size: int = 32
    lstm_hidden_size: int = 128
    num_to_predict: int = 1

    @nn.compact
    def __call__(self, observations, actions, training: bool = True):
        x = observations.reshape(-1, *observations.shape[2:])

        x = ResNet5(features=(64, 32), avg_pooling=False, flatten=True)(x, training=training)
        x = x.reshape(observations.shape[0], observations.shape[1], -1)  # (batch, num_datapoints, hidden_size)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        # x= nn.Dense(features=self.lstm_hidden_size)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.lstm_hidden_size)(x)
        # x = nn.relu(x)

        # add lstm
        # breakpoint()
        batch_size = observations.shape[0]
        lstm = SimpleScan(hidden_size=self.lstm_hidden_size)
        def init_carry(batch_size, hidden_size):
            return (jnp.zeros((batch_size, hidden_size)), jnp.zeros((batch_size, hidden_size)))
        init_carry = init_carry(batch_size, self.lstm_hidden_size)
        final_carry, x = lstm(init_carry, x)
        # x shape is now (batch, num_datapoints, lstm_hidden_size)

        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = x.reshape(x.shape[0] * x.shape[1], -1)  # Flatten
        x = nn.Dense(features=self.hidden_size)(x)
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
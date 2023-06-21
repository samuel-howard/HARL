"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_sr_base_runner import OnPolicySRBaseRunner

class OnPolicyHASRRunner(OnPolicySRBaseRunner):
    """Runner for on-policy HA sample reuse algorithms."""

    def train(self):
        """Train the model."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args['train']['episode_length']*self.algo_args['train']['M'],
                self.algo_args['train']['n_rollout_threads'],
                1,
            ),
            dtype=np.float32,
        )

        # Change advantage calculation for MARL V-trace.
        # This is to fix the advantage calculation, it doesn't affect the rest.
        if self.value_normalizer is not None:
            self.critic_buffer.returns[-1] = self.value_normalizer.denormalize(self.critic_buffer.value_preds[-1])*self.critic_buffer.masks[-1]
        else:
            self.critic_buffer.returns[-1] = self.critic_buffer.value_preds[-1]*self.critic_buffer.masks[-1]

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.clipped_rhos * (self.critic_buffer.rewards + 
                self.critic_buffer.gamma * self.critic_buffer.returns[1:] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            ) )
        else:
            advantages = self.critic_buffer.clipped_rhos * (self.critic_buffer.rewards + 
                self.critic_buffer.gamma * self.critic_buffer.returns[1:] - self.critic_buffer.value_preds[:-1])
            
        self.critic_buffer.returns[-1] = 0

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)


        # Calculate current log_probs (for pi_k)
        current_actions_logprob_agents = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                available_actions = (
                    None
                    if self.actor_buffer[agent_id].available_actions is None
                    else self.actor_buffer[agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
                    )
                
                # Obtain the pi_k log probs for the actions taken by the actor before the update
                current_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                    self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                    self.actor_buffer[agent_id].actions.reshape(
                        -1, *self.actor_buffer[agent_id].actions.shape[2:]
                    ),
                    self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                )
                current_actions_logprob_agents.append(current_actions_logprob)


        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(factor)  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # update actor
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), current_actions_logprob_agents[agent_id], "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), current_actions_logprob_agents[agent_id], "FP"
                ) # Sort later

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            old_actions_logprob = self.actor_buffer[agent_id].action_log_probs.reshape(new_actions_logprob.shape)
            old_actions_logprob = torch.Tensor(old_actions_logprob).to(self.actor[agent_id].device)
            # Have checked that this is the same as the above - though there seems to be some precision differences (gpu related)

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args['train']['episode_length']*self.algo_args['train']['M'],
                    self.algo_args['train']['n_rollout_threads'],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

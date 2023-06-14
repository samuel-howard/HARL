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



        # vs_t_plus_1 = tf.concat([
        # vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        # if clip_pg_rho_threshold is not None:
        # clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos,
        #                             name='clipped_pg_rhos')
        # else:
        # clipped_pg_rhos = rhos
        # pg_advantages = (
        #     clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))
    
        # compute advantages
        # if self.value_normalizer is not None:
        #     advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
        #         self.critic_buffer.value_preds[:-1]
        #     )
        # else:
        #     advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        # Change advantage calculation for MARL V-trace.

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.clipped_rhos * (self.critic_buffer.rewards + 
                self.critic_buffer.gamma * self.critic_buffer.returns[1:] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            ) )
        else:
            advantages = self.critic_buffer.clipped_rhos * (self.critic_buffer.rewards + 
                self.critic_buffer.gamma * self.critic_buffer.returns[1:] - self.critic_buffer.value_preds[:-1])

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

            # compute action log probs for the actor before update.
            # old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
            #     self.actor_buffer[agent_id]
            #     .obs[:-1]
            #     .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
            #     self.actor_buffer[agent_id]
            #     .rnn_states[0:1]
            #     .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
            #     self.actor_buffer[agent_id].actions.reshape(
            #         -1, *self.actor_buffer[agent_id].actions.shape[2:]
            #     ),
            #     self.actor_buffer[agent_id]
            #     .masks[:-1]
            #     .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
            #     available_actions,
            #     self.actor_buffer[agent_id]
            #     .active_masks[:-1]
            #     .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            # )

            # update actor
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                )

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

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args['train']['episode_length']*self.algo_args['train']['M'],            # FIX THIS - currently hard coded
                    self.algo_args['train']['n_rollout_threads'],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

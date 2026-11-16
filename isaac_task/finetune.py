"""Restore RSL policy/critic/Adam for a new, separately counted fine-tuning stage."""


def restore_finetune(runner, checkpoint):
    runner.load(str(checkpoint),load_optimizer=True,map_location=runner.device)
    parent_iteration=int(runner.current_learning_iteration)
    state=runner.alg.optimizer.state_dict()
    if not state['state']:
        raise ValueError('Fine-tuning requires a trained checkpoint with optimizer state')
    rates=[float(group['lr']) for group in state['param_groups']]
    if len(set(rates))!=1:
        raise ValueError('Expected one shared PPO optimizer learning rate')
    # RSL.load restores Adam's LR but not PPO.learning_rate. Keep the adaptive
    # scheduler consistent with the restored optimizer on the first new update.
    runner.alg.learning_rate=rates[0]
    runner.current_learning_iteration=0
    return dict(parent_iteration_zero_based=parent_iteration,parent_stage_iterations=parent_iteration+1,
                stage_iteration_start=0,optimizer_restored=True,optimizer_state_entries=len(state['state']),
                restored_learning_rate=rates[0],actor_critic_restored=True)

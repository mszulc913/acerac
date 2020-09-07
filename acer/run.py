import argparse

from runners import Runner, ALGOS

parser = argparse.ArgumentParser(
    description='Actor-Critic with experience replay algorithms.')
parser.add_argument(
    '--algo', type=str, help='Algorithm to be used (acer or acerac).', choices=ALGOS, required=True)
parser.add_argument(
    '--env_name', type=str, help='OpenAI Gym environment name.', default="CartPole-v0", required=True)
parser.add_argument(
    '--gamma', type=float, help='Discount factor.', required=False, default=0.99)
parser.add_argument(
    '--lam', type=float, help='Lambda parameter.', required=False, default=0.9)
parser.add_argument(
    '--b', type=float, help='Importance sampling truncation coefficient.', required=False, default=3)
parser.add_argument(
    '--actor_adam_epsilon', type=float, help='Epsilon for ADAM (Actor\'s network).',
    required=False, default=None)
parser.add_argument(
    '--actor_adam_beta1', type=float, help='Beta_1 for ADAM (Actor\'s network).',
    required=False, default=0.9)
parser.add_argument(
    '--actor_adam_beta2', type=float, help='Beta_2 for ADAM (Actor\'s network).',
    required=False, default=0.999)
parser.add_argument(
    '--critic_adam_epsilon', type=float, help='Epsilon for ADAM (Critic\'s network).',
    required=False, default=None)
parser.add_argument(
    '--critic_adam_beta1', type=float, help='Beta_1 for ADAM (Critic\'s network).',
    required=False, default=0.9)
parser.add_argument(
    '--critic_adam_beta2', type=float, help='Beta_2 for ADAM (Critic\'s network).',
    required=False, default=0.999)
parser.add_argument(
    '--actor_lr', type=float, help='Actor\'s step size.', required=False, default=3e-5)
parser.add_argument(
    '--critic_lr', type=float, help='Critic\'s step size', required=False, default=6e-5)
parser.add_argument(
    '--actor_beta_penalty', type=float, default=0.001,
    help='Penalty coefficient for Actor\'s actions (discrete case -- too confident actions, '
    'continuous case -- actions out of allowed boundaries')
parser.add_argument(
    '--c', type=int, help='Number of gradient steps per single update.', required=False, default=10)
parser.add_argument(
    '--learning_starts', type=int, help='Number of time steps without gradient steps.', default=1000)
parser.add_argument(
    '--alpha', type=float, help='ACERAC\'s alpha -- autocorrelation degree.', default=0.5)
parser.add_argument(
    '--tau', type=int, help='ACERAC\'s tau -- update window size.', default=4)
parser.add_argument(
    '--std', type=float, required=False, default=None,
    help='Value on diagonal of square root of Normal dist. covariance matrix.'
    'If not specified 0.4 * actions_bound is used.')
parser.add_argument(
    '--memory_size', type=int, help='Memory buffer size (sum of sizes of buffers from every parallel environment.)',
    required=False, default=1e6)
parser.add_argument(
    '--actor_layers', nargs='+', type=int, required=False, default=(256, 256),
    help='List of Actor\'s neural network hidden layers sizes.')
parser.add_argument(
    '--critic_layers', nargs='+', type=int, required=False, default=(256, 256),
    help='List of Critic\'s neural network hidden layers sizes')
parser.add_argument(
    '--num_parallel_envs', type=int, help='Number of environments to be run in a parallel.',
    default=1, required=False)
parser.add_argument(
    '--batches_per_env', type=int, default=256,
    help='Number of batches sampled from single environment\'s buffer. For num_parallel_envs == 1 it equals to the'
    'batch size.',
)
parser.add_argument(
    '--evaluate_time_steps_interval', type=int, default=30000,
    help='Number of time steps between evaluation runs, -1 to turn evaluation off')
parser.add_argument(
    '--num_evaluation_runs', type=int, help='Number of evaluation runs in a single evaluation.',
    default=10)
parser.add_argument(
    '--max_time_steps', type=int, help='Maximum number of time steps in a run. -1 means no time steps limit',
    default=-1)
parser.add_argument(
    '--log_dir', type=str, help='Logging directory.', default='logs/')
parser.add_argument(
    '--no_checkpoint', help='Disable checkpoint saving.', action='store_true')
parser.add_argument(
    '--no_tensorboard', help='Disable tensorboard logs.', action='store_true')
parser.add_argument(
    '--experiment_name', type=str, help='Name of the current experiment.', default='')
parser.add_argument(
    '--record_time_steps', type=int, default=None,
    help='Number of time steps between video recording.')
parser.add_argument(
    '--synchronous', action='store_true',
    help='True if not use asynchronous envs')


def main():
    args = parser.parse_args()

    cmd_parameters, unknown_args = parser.parse_known_args()
    if len(unknown_args):
        print("Not recognized arguments: ", str(vars(unknown_args)))
        return

    parameters = {k: v for k, v in vars(cmd_parameters).items() if v is not None}
    parameters.pop('env_name')
    evaluate_time_steps_interval = parameters.pop('evaluate_time_steps_interval')
    num_evaluation_runs = parameters.pop('num_evaluation_runs')
    max_time_steps = parameters.pop('max_time_steps')
    no_checkpoint = parameters.pop('no_checkpoint')
    no_tensorboard = parameters.pop('no_tensorboard')
    record_time_steps = parameters.pop('record_time_steps', None)
    experiment_name = parameters.pop('experiment_name')
    algorithm = parameters.pop('algo')
    log_dir = parameters.pop('log_dir')
    synchronous = parameters.pop('synchronous')

    runner = Runner(
        environment_name=cmd_parameters.env_name,
        algorithm=algorithm,
        algorithm_parameters=parameters,
        num_parallel_envs=cmd_parameters.num_parallel_envs,
        log_dir=log_dir,
        max_time_steps=max_time_steps,
        num_evaluation_runs=num_evaluation_runs,
        evaluate_time_steps_interval=evaluate_time_steps_interval,
        experiment_name=experiment_name,
        asynchronous=not synchronous,
        log_tensorboard=not no_tensorboard,
        do_checkpoint=not no_checkpoint,
        record_time_steps=record_time_steps
    )

    runner.run()


if __name__ == "__main__":
    main()

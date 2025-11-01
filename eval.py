import argparse

from src.utils.evaluate import eval_funcs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='one of star|agqa', required=True, choices=["agqa", "star"])
    parser.add_argument('--exp_name', help='experiment_name', required=True)
    parser.add_argument('--do_template',
                        default=False,
                        help='to do eval together based on `exp_name*` template; helpful for custom eval',
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    eval_funcs[args.dataset_name](args)

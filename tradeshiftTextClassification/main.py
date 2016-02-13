import regression as reg
import argparse

def add_arguments(parser):
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("-o", dest="out_file", required=True)
    parser.add_argument("--n_iter", required=True, type=int)
    parser.add_argument("-l", dest="learning_rate", required=True, type=float)
    parser.add_argument("--decay_learning_rate", dest="decay_rate", required=True, type=float)

    parser.add_argument("--batch_size", default=0.0, type=int)
    parser.add_argument("--holdout_size", default=0, type=float)
    parser.add_argument("--l2", default=0, type=float)
    parser.add_argument("--loss", default="squared")

    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--standardize", action="store_true", default=False)
    parser.add_argument("--adagrad", action="store_true", default=False)
    parser.add_argument("--rmsprop", action="store_true", default=False)
    parser.add_argument("--hash", default=0, type=int)

    parser.add_argument("--show", action="store_true", default=False)

parser = argparse.ArgumentParser()
add_arguments(parser)
opts = vars(parser.parse_args())

regression = reg.Regression(opts)
regression.learn()
regression.test()
regression.print_test_result()
if opts["show"]:
    regression.show_train_results()


from src.RKGE import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=50, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=5, help='embedding size')
    # parser.add_argument('--p', type=int, default=5, help='the number of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=10, help='embedding size')
    # parser.add_argument('--p', type=int, default=5, help='the number of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=10, help='embedding size')
    # parser.add_argument('--p', type=int, default=5, help='the number of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=20, help='embedding size')
    parser.add_argument('--p', type=int, default=5, help='the number of paths')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, True)

''''
music	train_auc: 0.862 	 train_acc: 0.827 	 eval_auc: 0.782 	 eval_acc: 0.755 	 test_auc: 0.778 	 test_acc: 0.752 		[0.01, 0.04, 0.21, 0.23, 0.23, 0.36, 0.41, 0.43]
book	train_auc: 0.757 	 train_acc: 0.700 	 eval_auc: 0.718 	 eval_acc: 0.695 	 test_auc: 0.723 	 test_acc: 0.694 		[0.11, 0.19, 0.32, 0.33, 0.33, 0.38, 0.39, 0.41]
ml	train_auc: 0.859 	 train_acc: 0.773 	 eval_auc: 0.852 	 eval_acc: 0.768 	 test_auc: 0.852 	 test_acc: 0.767 		[0.08, 0.16, 0.3, 0.32, 0.32, 0.41, 0.43, 0.45]
yelp	train_auc: 0.867 	 train_acc: 0.792 	 eval_auc: 0.843 	 eval_acc: 0.777 	 test_auc: 0.842 	 test_acc: 0.775 		[0.11, 0.22, 0.37, 0.41, 0.41, 0.46, 0.47, 0.47]

'''
from __future__ import print_function
import caffe
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--solver_state')

    args = parser.parse_args()
    caffe.set_mode_gpu()
    solver = caffe.get_solver(args.solver)
    if args.weights:
        solver.net.copy_from(args.weights)
    if args.solver_state:
        solver.restore(args.solver_state)
    solver.solve()

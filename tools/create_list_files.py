import sys
import os, os.path
import numpy as np
import argparse
import errno
from misc import make_sure_path_exists

                                
def read_scales(pose_file_path):
    poses = read_poses(pose_file_path)
    deltas = map(lambda prev_pose, pose: np.dot(np.linalg.inv(prev_pose), pose),
                 poses[:-1], poses[1:])
    scales = map(lambda delta: np.linalg.norm(delta[:3, 3]), deltas)
    return scales


def read_poses(pose_file_path):
    poses = []
    with open(pose_file_path, 'r') as f:
        for line in f:
            pose = np.array([float(val) for val in line.split()]).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    return poses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/data/other/')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--augment-same', action='store_true')
    parser.add_argument('--augment-reverse', action='store_true')
    parser.add_argument('--sha', default='')

    args = parser.parse_args()
    sequences = list(range(11))
    seq_dir = os.path.join(args.data_dir, 'dataset', 'sequences')
    poses_dir = os.path.join(args.data_dir, 'dataset', 'poses')

    scales = {s: read_scales(os.path.join(poses_dir, '%02d.txt' % s))
              for s in sequences}

    for camera in [['image_0'] ]:
        make_sure_path_exists(os.path.join(args.out_dir, '_'.join(camera)))
        for idx, test_seq in enumerate(sequences):
            train_seq = sequences[:idx] + sequences[idx+1:]
            train_file_name = os.path.join(args.out_dir, '_'.join(camera), 'no_%02d_%s.txt' % (test_seq, args.sha))
            print 'train_file_name', train_file_name
            with open(train_file_name, 'w+') as fd:
                for seq in train_seq:
                    for frame_no in range(len(scales[seq])):
                        for cam in camera:
                            name1 = os.path.join(seq_dir, '%02d' % seq, cam, '%06d.png' % frame_no)
                            name2 = os.path.join(seq_dir, '%02d' % seq, cam, '%06d.png' % (frame_no+1))
                            if os.path.exists(name1) and os.path.exists(name2):
                                print >>fd, name1, name2, scales[seq][frame_no]
                            if args.augment_same:
                                print >>fd, name1, name1, 0.
                            if args.augment_reverse:
                                print >>fd, name2, name1, scales[seq][frame_no]

            test_file_name = os.path.join(args.out_dir, '_'.join(camera), '%02d_%s.txt' % (test_seq, args.sha))
            print 'test_file_name', test_file_name
            with open(test_file_name, 'w+') as fd:
                for frame_no in range(len(scales[test_seq])):
                    for cam in camera:                        
                        name1 = os.path.join(seq_dir, '%02d' % test_seq, cam, '%06d.png' % frame_no)
                        name2 = os.path.join(seq_dir, '%02d' % test_seq, cam, '%06d.png' % (frame_no+1))
                        if os.path.exists(name1) and os.path.exists(name2):
                            print >>fd, name1, name2, scales[test_seq][frame_no]

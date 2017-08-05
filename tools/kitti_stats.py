from __future__ import print_function
import argparse
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', required=True)
    parser.add_argument('--out_file', default='a.png')
    
    args = parser.parse_args()

    with open(args.list_file, 'r') as fd:
        data = fd.read()

    data = filter(None, data.split('\n'))
    num_pairs = len(data)
    scales = []
    for x in data:
        scales.append(float(x.split()[2]))

    plt.hist(scales, bins=100, facecolor='green', alpha=0.75)
    plt.xlabel('Scale')
    plt.ylabel('Count')
    plt.title(r'$\mathrm{Histogram\ of\ scales\ %d\ data\ points}$' % num_pairs)
    plt.grid(True)
    plt.savefig(args.out_file, dpi=300)
    print('wrote ', args.out_file)

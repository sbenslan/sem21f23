import argparse
import horovod.torch as hvd

from experiments import Logbook
from experiments import get_data, get_network, get_training
from experiments import train, validate


def main():

    # Command Line Interface
    parser = argparse.ArgumentParser(description='QuantLab')
    parser.add_argument('--mode',       help='Mode: train/test/demo/deploy',           default='train')
    parser.add_argument('--ckpt_every', help='Frequency of checkpoints (in epochs)',   default=5)
    parser.add_argument('--problem',    help='Data set')
    parser.add_argument('--topology',   help='Network topology')
    parser.add_argument('--exp_id',     help='Experiment to launch/resume/test',       default=None)
    parser.add_argument('--i_fold',     help='Cross-validation fold (for checkpoint)', default=None)
    parser.add_argument('--ckpt_name',  help='Checkpoint to test/demo/deploy',         default=None)
    parser.add_argument('--backend',    help='Target backend hardware platform',       default=None)
    args = parser.parse_args()

    # initialise Horovod
    hvd.init()

    # create/retrieve experiment logbook
    logbook = Logbook(args.problem, args.topology, args.exp_id)

    # run experiment
    if args.mode == 'train':

        logbook.get_training_status()

        for i_fold in range(logbook.i_fold, logbook.config['experiment']['n_folds']):

            # create data sets, network, and training algorithm
            train_l, valid_l = get_data(logbook)
            net = get_network(logbook)
            loss_fn, opt, lr_sched, ctrls = get_training(logbook, net)

            # boot logging instrumentation and load most recent checkpoint
            logbook.open_fold()

            logbook.load_checkpoint(net, opt, lr_sched, ctrls, 'last')

            # main routine
            for i_epoch in range(logbook.i_epoch, logbook.config['experiment']['n_epochs']):

                train_stats = train(logbook, train_l, net, loss_fn, opt, ctrls)
                valid_stats = validate(logbook, valid_l, net, loss_fn, ctrls)

                # (maybe) update learning rate
                stats = {**train_stats, **valid_stats}
                if 'metrics' in lr_sched.step.__code__.co_varnames:
                    lr_sched_metric = stats[logbook.config['training']['lr_scheduler']['step_metric']]
                    lr_sched.step(lr_sched_metric)
                else:
                    lr_sched.step()

                # save model if last/checkpoint epoch, and/or if update metric has improved
                is_last_epoch = (logbook.i_epoch + 1) == logbook.config['experiment']['n_epochs']
                is_ckpt_epoch = ((logbook.i_epoch + 1) % args.ckpt_every) == 0
                is_best = logbook.is_better(stats)
                if is_last_epoch or is_ckpt_epoch:
                    logbook.store_checkpoint(net, opt, lr_sched, ctrls)
                if is_best:
                    logbook.store_checkpoint(net, opt, lr_sched, ctrls, is_best=True)

                logbook.i_epoch += 1

            logbook.close_fold()

    # elif args.mode == 'test':
    #     # test
    #     net.eval()
    #     test_stats = test(logbook, net, device, loss_fn, test_l)

    elif args.mode == 'deploy':

        import importlib
        export_lib = importlib.import_module('.'.join(['backends', args.backend]))
        export_fun = getattr(export_lib, 'export')

        export_fun(args.problem, args.exp_id, args.i_fold, args.ckpt_name)


if __name__ == '__main__':
    main()

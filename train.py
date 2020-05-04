import argparse

def get_params():
  """
  Parse args params
  :return:
  """
  parser = argparse.ArgumentParser()

  # data hyperparams
  parser.add_argument("--input_size", type=int, default=10)

  parser.add_argument("--framework", type=str, default="sklearn")
  parser.add_argument("--keras_model", type=str, default="dense")
  parser.add_argument("--sklearn_model", type=str, default="linear")
  parser.add_argument("--loss", type=str, default="squared_loss")

  # training hyperparams

  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--epoch", default=10, type=int)
  parser.add_argument("--lr", default=0.001, type=float)

  args = parser.parse_args()

  return args


def hyperparam():
  """
  sacred exmperiment hyperparams
  :return:
  """

  args = get_params()

  args.framework = args.framework
  args.sklearn_model = args.sklearn_model
  args.loss = args.loss
  args.batch_size = args.batch_size
  args.fc1_size = args.fc1_size
  args.epoch = args.epoch
  args.lr = args.lr

  print("hyperparam - ", args)


def run(args):
  """
  Run sacred experiment
  :param args:
  :return:
  """
  # if args.framework == 'sklearn':
  #   test_loss = sk.train_sklearn(args)
  # elif args.framework == 'keras':
  #   test_loss = tf2.train_keras(args, ex)
  # else:
  #   return None

  # return test_loss

__author__ = 'coenjonker'

from argparse import ArgumentParser

from primitive.functions import read_data, write_data


if __name__ == "__main__":

    parser = ArgumentParser(description="Train a model and save predicted classes on test set")
    parser.add_argument("train", type=str, help='csv file met test data')
    parser.add_argument("test", type=str, help='csv file contiaing test data')
    parser.add_argument("outputfile", type=str, help='file name to write output to (for submission to Kaggle)')
    parser.add_argument("--model", type=str, help="model to run", default="MostLikelyClassModel")

    args = parser.parse_args()

    model = eval(args.model)()

    model.train(read_data(args.train))
    write_data(args.outputfile, model, args.test)














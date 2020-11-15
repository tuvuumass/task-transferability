# coding: utf8
def main():
    import sys
    if (len(sys.argv) < 4 or len(sys.argv) > 6) or sys.argv[1] not in ["bert"]:
        print(
        "This command line utility let you convert original (author released) model checkpoint to pytorch.\n"
        "It should be used as one of: \n"
        ">> transformers bert TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT, \n")
    else:
        if sys.argv[1] == "bert":
            try:
                from .convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
            except ImportError:
                print("transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                    "In that case, it requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
                raise

            if len(sys.argv) != 5:
                # pylint: disable=line-too-long
                print("Should be used as `transformers bert TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT`")
            else:
                PYTORCH_DUMP_OUTPUT = sys.argv.pop()
                TF_CONFIG = sys.argv.pop()
                TF_CHECKPOINT = sys.argv.pop()
                convert_tf_checkpoint_to_pytorch(TF_CHECKPOINT, TF_CONFIG, PYTORCH_DUMP_OUTPUT)


if __name__ == '__main__':
    main()

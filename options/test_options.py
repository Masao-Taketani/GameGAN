from argparse import ArgumentParser


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.set_opts()

    def set_opts(self):
        self.parser.add_argument('--model_path', type=str, help='model path to run a test')
        self.parser.add_argument('--init_img_path', type=str, help='image path to initialize the similator')
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts
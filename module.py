class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        print("hi, I am the main module")
        raise NotImplementedError

    def get_parameters(self):
        return []

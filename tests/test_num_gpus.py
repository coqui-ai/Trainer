import os
from trainer.distribute import get_gpus
from trainer import TrainerArgs
from unittest import TestCase, mock


class TestStringMethods(TestCase):

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
    def test_parse_gpus_set_in_env_var_and_args(self):
        parsed_args = create_args_parser().parse_args(['--gpus', '0,1'])
        gpus = get_gpus(parsed_args)
        expected_value = ['0']
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))

    @mock.patch.dict(os.environ, {})
    def test_parse_gpus_set_in_args(self):
        parsed_args = create_args_parser().parse_args(['--gpus', '0,1'])
        gpus = get_gpus(parsed_args)
        expected_value = ['0', '1']
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
    def test_parse_gpus_set_in_env_var(self):
        parsed_args = create_args_parser().parse_args(None)
        gpus = get_gpus(parsed_args)
        expected_value = ['0', '1']
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0, 1 "})
    def test_parse_gpus_set_in_env_var_with_spaces(self):
        parsed_args = create_args_parser().parse_args(None)
        gpus = get_gpus(parsed_args)
        expected_value = ['0', '1']
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))

    @mock.patch.dict(os.environ, {})
    def test_parse_gpus_set_in_args_with_spaces(self):
        parsed_args = create_args_parser().parse_args(['--gpus', '0, 1, 2, 3 '])
        gpus = get_gpus(parsed_args)
        expected_value = ['0', '1', '2', '3']
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))


def msg_for_test_failure(expected_value):
    return "GPU Values are expected to be " + str(expected_value)


def create_args_parser():
    parser = TrainerArgs().init_argparse(arg_prefix="")
    parser.add_argument("--gpus", type=str)
    return parser


if __name__ == '__main__':
    unittest.main()
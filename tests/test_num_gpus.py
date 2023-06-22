import os
import unittest
from argparse import Namespace
from unittest import TestCase, mock

from trainer import TrainerArgs
from trainer.distribute import get_gpus


class TestGpusStringParsingMethods(TestCase):
    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
    def test_parse_gpus_set_in_env_var_and_args(self):
        args = Namespace(gpus="0,1")
        gpus = get_gpus(args)
        expected_value = ["0"]
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))

    @mock.patch.dict(os.environ, {})
    def test_parse_gpus_set_in_args(self):
        _old = None
        # this is to handle the case when CUDA_VISIBLE_DEVICES is set while running the tests
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            _old = os.environ["CUDA_VISIBLE_DEVICES"]
            del os.environ["CUDA_VISIBLE_DEVICES"]
        args = Namespace(gpus="0,1")
        gpus = get_gpus(args)
        expected_value = ["0", "1"]
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))
        if _old is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _old

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
    def test_parse_gpus_set_in_env_var(self):
        args = Namespace()
        gpus = get_gpus(args)
        expected_value = ["0", "1"]
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))

    @mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0, 1 "})
    def test_parse_gpus_set_in_env_var_with_spaces(self):
        args = Namespace()
        gpus = get_gpus(args)
        expected_value = ["0", "1"]
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))

    @mock.patch.dict(os.environ, {})
    def test_parse_gpus_set_in_args_with_spaces(self):
        args = Namespace(gpus="0, 1, 2, 3 ")
        gpus = get_gpus(args)
        expected_value = ["0", "1", "2", "3"]
        self.assertEqual(expected_value, gpus, msg_for_test_failure(expected_value))


def msg_for_test_failure(expected_value):
    return "GPU Values are expected to be " + str(expected_value)


def create_args_parser():
    parser = TrainerArgs().init_argparse(arg_prefix="")
    parser.add_argument("--gpus", type=str)
    return parser


if __name__ == "__main__":
    unittest.main()

"""
Contains the serializers for the ReturnnConfig epilog which write the model import and task functions import
including serializing their parameters.
"""
from typing import Any, Dict, Optional

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import ExternalImport, Import, PartialImport

from . import PACKAGE
from .default_tools import I6_MODELS_REPO_PATH, TORCH_MEMRISTOR_PATH


def serialize_training(
    network_module: str,
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    import_memristor: bool = False,
) -> Collection:
    """
    Helper function to create the serialization collection

    :param network_module: path to the pytorch config file containing Model
    :param net_args: arguments for the model
    :param unhashed_net_args: as above but not hashed
    :param debug: run training in debug mode: linking from recipe instead of copy
    :return: Collection object to be added to the ReturnnConfig epilog
    """
    package = PACKAGE + ".pytorch_networks"

    pytorch_model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )
    pytorch_train_step = Import(
        code_object_path=package + ".%s.train_step" % network_module, unhashed_package_root=PACKAGE
    )
    i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)

    serializer_objects = [
        i6_models,
        pytorch_model_import,
        pytorch_train_step,
    ]
    if import_memristor is True:
        memristor_modules = ExternalImport(import_path=TORCH_MEMRISTOR_PATH)
        # serializer_objects.insert(1, memristor_modules)
        serializer_objects.append(memristor_modules)

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )
    return serializer


def serialize_forward(
    network_module: str,
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    forward_module: Optional[str] = None,
    forward_step_name: str = "forward",
    forward_init_args: Optional[Dict[str, Any]] = None,
    unhashed_forward_init_args: Optional[Dict[str, Any]] = None,
    import_memristor: bool = False,
    debug: bool = False,
):
    """
    Serialize for a forward job. Can be used e.g. for search or prior computation.

    :param network_module: path to the pytorch config file containing Model
    :param net_args: arguments for the model
    :param unhashed_net_args: as above but not hashed
    :param forward_module: optionally define a module file which contains the forward definition.
        If not provided the network_module is used.
    :param forward_step_name: path to the search decoder file containing forward_step and hooks
    :param forward_init_args: additional arguments to pass to forward_init
    :param unhashed_forward_init_args: additional non-hashed arguments to pass to forward_init
    :param debug: run training in debug mode: linking from recipe instead of copy
    :return:
    """

    package = PACKAGE + ".pytorch_networks"
    pytorch_model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )

    i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)

    serializer_objects = [
        i6_models,
        pytorch_model_import,
    ]
    if import_memristor is True:
        memristor_modules = ExternalImport(import_path=TORCH_MEMRISTOR_PATH)
        serializer_objects.insert(1, memristor_modules)

    forward_module = forward_module or network_module

    forward_step = Import(
        code_object_path=package + ".%s.%s_step" % (forward_module, forward_step_name),
        unhashed_package_root=PACKAGE,
        import_as="forward_step",
    )
    init_hook = PartialImport(
        code_object_path=package + ".%s.%s_init_hook" % (forward_module, forward_step_name),
        unhashed_package_root=PACKAGE,
        hashed_arguments=forward_init_args or {},
        unhashed_arguments=unhashed_forward_init_args or {},
        import_as="forward_init_hook",
    )
    finish_hook = Import(
        code_object_path=package + ".%s.%s_finish_hook" % (forward_module, forward_step_name),
        unhashed_package_root=PACKAGE,
        import_as="forward_finish_hook",
    )
    serializer_objects.extend([forward_step, init_hook, finish_hook])

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )
    return serializer


def serialize_quant(
    network_module: str,
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    export_module: Optional[str] = None,
    export_step_name: str = "forward",
    debug: bool = False,
):
    """
    Serialize for a forward job. Can be used e.g. for search or prior computation.

    :param network_module: path to the pytorch config file containing Model
    :param net_args: arguments for the model
    :param unhashed_net_args: as above but not hashed
    :param forward_module: optionally define a module file which contains the forward definition.
        If not provided the network_module is used.
    :param forward_step_name: path to the search decoder file containing forward_step and hooks
    :param forward_init_args: additional arguments to pass to forward_init
    :param unhashed_forward_init_args: additional non-hashed arguments to pass to forward_init
    :param debug: run training in debug mode: linking from recipe instead of copy
    :return:
    """

    package = PACKAGE + ".pytorch_networks"

    pytorch_model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )

    i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)

    serializer_objects = [
        i6_models,
        pytorch_model_import,
    ]

    forward_module = export_module or network_module

    forward_step = Import(
        code_object_path=package + ".%s.%s" % (forward_module, export_step_name),
        unhashed_package_root=PACKAGE,
        import_as="export",
    )
    serializer_objects.extend([forward_step])

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )
    return serializer

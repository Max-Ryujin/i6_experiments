import torch
from returnn.tensor.tensor_dict import TensorDict


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **_):
    audio_features = extern_data["data"].raw_tensor.float()
    audio_features = audio_features.squeeze(-1)
    assert extern_data["data"].dims[1].dyn_size_ext is not None

    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    assert extern_data["classes"].raw_tensor is not None
    targets = extern_data["classes"].raw_tensor.long()

    targets_len_rf = extern_data["classes"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None

    log_probs, sequence_lengths = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]

    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=sequence_lengths,
        target_lengths=targets_len,
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )

    import returnn.frontend as rf
    from returnn.tensor import batch_dim

    rf.get_run_ctx().mark_as_loss(
        name="CTC", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )

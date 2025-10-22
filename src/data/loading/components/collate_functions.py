from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.loading.components.interfaces import (
    LabelFunctionOutput,
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.data.loading.utils import combine_list_of_tensor_dicts, pad_or_trim_sequence
from src.utils.tensor_utils import extract_locations
from src.data.loading.components.interfaces import ItemData

def identity_collate_fn(batch: Any) -> Any:
    """The default collate function that does nothing."""
    return batch


def collate_with_sid_causal_duplicate(
    # batch can be a list or a dict
    # this function is used to create the generate contiguous sequences as data augmentation to improve the performance
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    sequence_field_name: str,
    sid_hierarchy: int,
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    max_batch_size: int = 128,
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """
        this collate fn is used to create the generate contiguous sequences as data augmentation to improve the performance.
        It does three things
        1. augment the input sequences by creating all possible contiguous sequences
        2. random sample max_batch_size sequences from the augmented sequences to prevent OOM
        3. run regular collate_fn_train
    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    sequence_field_name : str
        The name of the field in the batch that contains the sequence to be augmented.
    sid_hierarchy : int
        The length of Semantic IDs
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels.
    sequence_length : int
        The length of the sequence to be padded or trimmed to. (not used in this function, passed to collate_fn_train)
    masking_token : int
        The token used for masking. (not used in this function, passed to collate_fn_train)
    padding_token : int
        The token used for padding. (not used in this function, passed to collate_fn_train)
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence. (not used in this function, passed to collate_fn_train)
    max_batch_size : int
        The maximum batch size to be used after the data augmentation.
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore

    # calculating the total number of contiguous sub-sequences in the batch
    total_num_seqs = torch.sum(
        (
            (
                k := torch.tensor([s.shape[0] for s in batch[sequence_field_name]])
                // sid_hierarchy
            )
            - 1
        )
        * k
        // 2
    )

    if total_num_seqs > max_batch_size:
        select_seqs = torch.randint(
            low=0,
            high=total_num_seqs,
            size=(max_batch_size,),
        )
    else:
        select_seqs = torch.arange(total_num_seqs)

    new_batch = {field_name: [] for field_name in batch}
    current_idx = 0
    for row_index, sequence in enumerate(batch[sequence_field_name]):
        end_indices = torch.arange(
            2 * sid_hierarchy, sequence.shape[0] + 1, sid_hierarchy
        )
        for end_index in end_indices:
            start_indices = torch.arange(
                0, end_index - 2 * sid_hierarchy + 1, sid_hierarchy
            )  # we have a -2 here because we want to have at least two items in the sequence
            for start_index in start_indices:
                if current_idx in select_seqs:
                    new_batch[sequence_field_name].append(
                        sequence[start_index:end_index]
                    )
                    for field_name in new_batch:
                        if field_name != sequence_field_name:
                            new_batch[field_name].append(batch[field_name][row_index])
                current_idx += 1

    return collate_fn_train(
        batch=new_batch,
        labels=labels,
        sequence_length=sequence_length,
        masking_token=masking_token,
        padding_token=padding_token,
        oov_token=oov_token,
    )


def collate_fn_inference_for_sequence(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    id_field_name: str,
    sequence_length: int = 200,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    **kwargs,
) -> SequentialModelInputData:
    """The collate function passed to inference dataloader for inference with sequential data.
    It handles id_field_name for saving model outputs

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    sequence_length : int
        The length of the sequence to be padded or trimmed to.
    masking_token : int
        The token used for masking.
    padding_token : int
        The token used for padding.
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence.
    id_field_name : str
        The name of the field that contains the id of the user/item. This is used to
        map the predictions back to the original id.
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore

    model_input_data = SequentialModelInputData()

    for field_name, field_sequence in batch.items():  # type: ignore
        if field_name in id_field_name:
            # We use the id field as the user_id_list so predictions can be mapped back to the original id.
            model_input_data.user_id_list = field_sequence

        # TODO (lneves): Allow for non-sequential data to be passed as a feature.
        current_sequence = field_sequence  # type: ignore
        if oov_token:
            # removing the oov token # TODO (Clark): in the future we can add special OOV handling
            current_sequence = [
                sequence[sequence != oov_token] for sequence in field_sequence
            ]
        # 1. in-batch padding s.t. all sequences have the same length and in the format of pt tensor
        current_sequence = pad_sequence(
            current_sequence, batch_first=True, padding_value=padding_token
        )

        # 2. padding or trimming the sequence to the desired length for training
        current_sequence = pad_or_trim_sequence(
            padded_sequence=current_sequence,
            sequence_length=sequence_length,
            padding_token=padding_token,
        )
        model_input_data.transformed_sequences[field_name] = current_sequence

        if field_name not in id_field_name and model_input_data.mask is None:
            # if a field is not id, then it means its the real sequence we want calculate attention mask for it
            model_input_data.mask = (current_sequence != padding_token).long()

    return model_input_data  # type: ignore


def collate_fn_train(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    data_augmentation_functions: Optional[
        List[Dict[str, callable]]
    ] = None,  # type: ignore
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """The collate function passed to dataloader. It can do training masking and padding for the input sequence.

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels.
    sequence_length : int
        The length of the sequence to be padded or trimmed to.
    masking_token : int
        The token used for masking.
    padding_token : int
        The token used for padding.
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence.
    data_augmentation_functions : Optional[List[Dict[str, callable]]]
        The list of functions to apply to augment the data.
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore

    if data_augmentation_functions:
        for data_augmentation_function in data_augmentation_functions:
            batch = data_augmentation_function(batch)

    model_input_data = SequentialModelInputData()
    model_label_data = SequentialModuleLabelData()

    for field_name, field_sequence in batch.items():  # type: ignore
        # TODO (lneves): Allow for non-sequential data to be passed as a feature.
        current_sequence = field_sequence  # type: ignore
        if oov_token:
            # removing the oov token # TODO (Clark): in the future we can add special OOV handling
            current_sequence = [
                sequence[sequence != oov_token] for sequence in field_sequence
            ]
        # 1. in-batch padding s.t. all sequences have the same length and in the format of pt tensor
        current_sequence = pad_sequence(
            current_sequence, batch_first=True, padding_value=padding_token
        )

        # 2. padding or trimming the sequence to the desired length for training
        current_sequence = pad_or_trim_sequence(
            padded_sequence=current_sequence,
            sequence_length=sequence_length,
            padding_token=padding_token,
        )

        # creating labels if the field is in the labels list
        if field_name in labels:
            label_function = labels[field_name].transform
            label_function_output: LabelFunctionOutput = label_function.transform_label(
                sequence=current_sequence,
                padding_token=padding_token,
                masking_token=masking_token,
            )
            model_label_data.labels[field_name] = label_function_output.labels
            model_label_data.label_location[
                field_name
            ] = label_function_output.label_location
            model_label_data.attention_mask[
                field_name
            ] = label_function_output.attention_mask
            model_input_data.transformed_sequences[
                field_name
            ] = label_function_output.sequence
        else:
            model_input_data.transformed_sequences[field_name] = current_sequence

        # Currently supports a single masking per sequence
        # TODO (lneves): Evaluate if this works or if we should have one mask per feature.
        if model_input_data.mask is None:
            model_input_data.mask = (current_sequence != padding_token).long()

    return model_input_data, model_label_data  # type: ignore


def collate_fn_items(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    item_id_field: str,
    feature_to_input_name: Dict[str, str],  # type: ignore
) -> ItemData:
    """The collate function passed to the item dataloader.

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we
        loaded the data per row, or a dictionary of tensors, in the case loaded the data
        per batch.
    item_id_field : str
        The name of the field in the batch that contains the item IDs.
    feature_to_input_name : Dict[str, str]
        The mapping from raw feature name to input feature name in ItemData.

    Returns:
    --------
    model_input_data : ItemData
        An ItemData object, which stores a batch of item features via a list of item IDs
        in the field `item_ids` and a dictionary mapping feature names to value tensors
        stacked along the batch dimension.
    """
    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore
        # does not change shape of text tokens

    model_input_data = ItemData()

    for field_name, field_value in batch.items():  # type: ignore
        if field_name == item_id_field:
            model_input_data.item_ids = list(field_value)

        else:
            # In this case, field_value is a list of tensors, each representing the
            # features of a single item. We stack these tensors along the batch
            # dimension to create a single tensor for the batch of items.
            field_value = torch.stack(field_value, dim=0)
            model_input_data.transformed_features[
                feature_to_input_name[field_name]
            ] = field_value

    return model_input_data
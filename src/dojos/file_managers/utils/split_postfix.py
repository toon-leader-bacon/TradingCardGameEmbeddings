def get_split_file_postfix(split_index: int) -> str:
    """Format-agnostic split-index-to-filename-postfix mapping, shared by
    every FileManager* implementation (0=train, 1=test, 2=validation,
    else split_{index}).
    """
    if split_index == 0:
        return "train"
    elif split_index == 1:
        return "test"
    elif split_index == 2:
        return "validation"
    else:
        return f"split_{split_index}"

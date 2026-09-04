# TODO

- Make a FileManagerJsonL.py
- Consider deduplicating between these types (FileManagerCSV still has its
  own private _get_file_postfix instead of using the shared
  utils/split_postfix.get_split_file_postfix that FileManagerParquet uses)

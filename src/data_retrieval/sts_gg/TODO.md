=== sts2runs ===
Traceback (most recent call last):
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\connection.py", line 204, in _new_conn
    sock = connection.create_connection(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\util\connection.py", line 60, in create_connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ender\AppData\Local\Programs\Python\Python312\Lib\socket.py", line 978, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
socket.gaierror: [Errno 11001] getaddrinfo failed

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\connectionpool.py", line 788, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\connectionpool.py", line 488, in _make_request
    raise new_e
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\connectionpool.py", line 464, in _make_request
    self._validate_conn(conn)
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\connectionpool.py", line 1106, in _validate_conn
    conn.connect()
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\connection.py", line 759, in connect
    self.sock = sock = self._new_conn()
                       ^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\connection.py", line 211, in _new_conn
    raise NameResolutionError(self.host, self, e) from e
urllib3.exceptions.NameResolutionError: HTTPSConnection(host='sts2runs.com', port=443): Failed to resolve 'sts2runs.com' ([Errno 11001] getaddrinfo failed)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\requests\adapters.py", line 696, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\connectionpool.py", line 842, in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\urllib3\util\retry.py", line 543, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='sts2runs.com', port=443): Max retries exceeded with url: /downloads/runs-all-before-2026-06.json.gz (Caused by NameResolutionError("HTTPSConnection(host='sts2runs.com', port=443): Failed to resolve 'sts2runs.com' ([Errno 11001] getaddrinfo failed)"))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "G:\Projects\TradingCardGameEmbeddings\scripts\run_data_retrieval.py", line 268, in run_one
    DOWNLOADERS[name]()
  File "G:\Projects\TradingCardGameEmbeddings\scripts\run_data_retrieval.py", line 152, in run_sts2runs
    _run_standard(STS2RunsDownloader())
  File "G:\Projects\TradingCardGameEmbeddings\scripts\run_data_retrieval.py", line 116, in _run_standard
    downloader.phase_1()
  File "G:\Projects\TradingCardGameEmbeddings\src\data_retrieval\downloader.py", line 100, in phase_1
    self._phase_1_result = self._run_phase_1()
                           ^^^^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\src\data_retrieval\sts2runs\downloader.py", line 108, in _run_phase_1
    compressed_path = self.download()
                      ^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\src\data_retrieval\sts2runs\downloader.py", line 128, in download
    download_to_file(self.source_url, destination_path)
  File "G:\Projects\TradingCardGameEmbeddings\src\data_retrieval\download_utils.py", line 174, in download_to_file
    call_with_retries(_attempt, max_attempts=max_attempts)
  File "G:\Projects\TradingCardGameEmbeddings\src\data_retrieval\download_utils.py", line 114, in call_with_retries
    raise last_error
  File "G:\Projects\TradingCardGameEmbeddings\src\data_retrieval\download_utils.py", line 107, in call_with_retries
    return attempt()
           ^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\src\data_retrieval\download_utils.py", line 163, in _attempt
    with requests.get(
         ^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\requests\api.py", line 87, in get
    return request("get", url, params=params, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\requests\api.py", line 71, in request
    return session.request(method=method, url=url, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\requests\sessions.py", line 651, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\requests\sessions.py", line 784, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\Projects\TradingCardGameEmbeddings\venv\Lib\site-packages\requests\adapters.py", line 729, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='sts2runs.com', port=443): Max retries exceeded with url: /downloads/runs-all-before-2026-06.json.gz (Caused by NameResolutionError("HTTPSConnection(host='sts2runs.com', port=443): Failed to resolve 'sts2runs.com' ([Errno 11001] getaddrinfo failed)"))

Press Enter to close this window...
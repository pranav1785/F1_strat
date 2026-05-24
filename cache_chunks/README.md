# F1 Cache Chunks

`f1_cache.zip` and `f1_cache/fastf1_http_cache.sqlite` are split into GitHub-safe chunks because the original files are larger than GitHub's 100 MB single-file limit.

Rebuild the archive and SQLite cache from PowerShell:

```powershell
$parts = Get-ChildItem .\cache_chunks\f1_cache.zip.part* | Sort-Object Name
$out = [System.IO.File]::Create("f1_cache.zip")
try {
  foreach ($part in $parts) {
    $bytes = [System.IO.File]::ReadAllBytes($part.FullName)
    $out.Write($bytes, 0, $bytes.Length)
  }
}
finally {
  $out.Dispose()
}

$sqliteParts = Get-ChildItem .\cache_chunks\fastf1_http_cache.sqlite.part* | Sort-Object Name
$sqliteOut = [System.IO.File]::Create("f1_cache\fastf1_http_cache.sqlite")
try {
  foreach ($part in $sqliteParts) {
    $bytes = [System.IO.File]::ReadAllBytes($part.FullName)
    $sqliteOut.Write($bytes, 0, $bytes.Length)
  }
}
finally {
  $sqliteOut.Dispose()
}
```

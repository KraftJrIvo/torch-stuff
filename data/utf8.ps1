# for .Net methods, you need absolute paths
$file = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($pwd, '.\russian_surnames.txt'))

# get the 1251 encoding object
$cyrillicEnc = [System.Text.Encoding]::GetEncoding("windows-1251")
# read all lines in an array
$contents = [System.IO.File]::ReadAllLines($file, $cyrillicEnc)

$OutFile   = 'res.txt'
$Stream = [System.IO.StreamWriter]::new($OutFile)
foreach ($line in $contents) {
    $Stream.WriteLine($line)
}
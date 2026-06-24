import tarfile

# Extract all contents to current directory
with tarfile.open('lee.tar.gz', 'r') as tar:
    tar.extractall()

# Extract to a specific directory
with tarfile.open('lee.tar.gz', 'r') as tar:
    tar.extractall(path='extracted_folder')

# List contents first (optional)
with tarfile.open('lee.tar.gz', 'r') as tar:
    print("Contents of the tar file:")
    print(tar.getnames())
    tar.extractall()
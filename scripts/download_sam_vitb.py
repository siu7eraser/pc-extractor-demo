import urllib.request
import os

url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
dest = 'weights/sam_vit_b_01ec64.pth'

os.makedirs('weights', exist_ok=True)

print('Downloading SAM ViT-B model (375MB)...')
print(f'URL: {url}')
print(f'Dest: {dest}')

def reporthook(count, block_size, total_size):
    percent = min(int(count * block_size * 100 / total_size), 100)
    mb_downloaded = count * block_size / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f'\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='')

urllib.request.urlretrieve(url, dest, reporthook)
print('\nDownload complete!')
print(f'File saved to: {dest}')

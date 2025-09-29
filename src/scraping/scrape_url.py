from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import pandas as pd
import time
import os
import random
from datetime import timedelta, datetime

# Inisialisasi WebDriver menggunakan WebDriver Manager
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# URL awal
base_url = "https://www.rumah123.com/jual/cari/?location=tangerang&page=1"

# Tunggu halaman termuat
wait = WebDriverWait(driver, 10)

# Nama file output CSV
output_file = "filtered_links.csv"

# Jika file CSV sudah ada, baca data lama
if os.path.exists(output_file):
    existing_links = pd.read_csv(output_file)["URL"].tolist()
else:
    existing_links = []

# Daftar untuk menyimpan semua link baru
filtered_links = []

# Variabel untuk membatasi scraping
start_page = 501  # Halaman awal yang ingin di-scrape
end_page = 600    # Halaman akhir yang ingin di-scrape

# Mulai timer
start_time = datetime.now()

try:
    with tqdm(total=(end_page - start_page + 1), desc="Scraping Progress", unit="halaman") as pbar:
        page_counter = start_page
        while page_counter <= end_page:
            print(f"Scraping halaman {page_counter}...")

            # Akses halaman yang sesuai
            current_url = f"https://www.rumah123.com/jual/cari/?location=tangerang&page={page_counter}"
            driver.get(current_url)

            # Tunggu elemen properti muncul
            properties = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[title]')))

            # Ambil link dari tiap properti di halaman
            for property_elem in properties:
                link = property_elem.get_attribute('href')
                if 'hos' in link and 'properti' in link and 'perumahan-baru' not in link and link not in existing_links and link not in filtered_links:
                    filtered_links.append(link)

            # Cek apakah sudah mencapai batas halaman
            if page_counter == end_page:
                print(f"Telah mencapai batas halaman {end_page}. Scraping selesai.")
                break

            page_counter += 1  # Tambah penghitung halaman
            pbar.update(1)  # Update progress bar

            # Tambahkan jeda acak antara 2 hingga 15 detik
            time.sleep(random.uniform(2, 15))

finally:
    # Tutup browser
    driver.quit()

# Hitung durasi scraping
end_time = datetime.now()
duration = end_time - start_time
formatted_duration = str(timedelta(seconds=int(duration.total_seconds())))
print(f"Durasi scraping: {formatted_duration}")

# Simpan hasil ke CSV
if filtered_links:
    all_links = existing_links + filtered_links
    df = pd.DataFrame(all_links, columns=["URL"])
    df.to_csv(output_file, index=False)
    print(f"Scraping selesai. {len(filtered_links)} link baru ditambahkan. Total {len(all_links)} link disimpan dalam '{output_file}'.")
else:
    print("Tidak ada link baru yang ditemukan.")
import pandas as pd
import requests
from bs4 import BeautifulSoup
import random
import time
import re
from tqdm import tqdm
from datetime import timedelta
import os
import winsound

# Baca URL dari file CSV
input_file = "filtered_links.csv"
urls = pd.read_csv(input_file)["URL"].tolist()
start_index = 7089  # Indeks awal untuk scraping
urls = urls[start_index:-1]  # Scrape sampai URL ke-n

# Nama file output CSV
output_file = "hasil_scraping_rumah123.csv"

# Jika file hasil scraping sudah ada, baca URL yang sudah di-scrape
if os.path.exists(output_file):
    existing_data = pd.read_csv(output_file)
    scraped_urls = existing_data["URL"].tolist()
else:
    scraped_urls = []

# Filter URL yang belum di-scrape
urls_to_scrape = [url for url in urls if url not in scraped_urls]

# Load User-Agent dari file
user_agent_file = "user_agents.txt"
with open(user_agent_file, "r") as f:
    user_agents = [line.strip() for line in f if line.strip()]

used_user_agents = []  # Daftar User-Agent yang sudah digunakan

# Fungsi untuk memilih User-Agent secara random dengan aturan
def get_random_user_agent():
    global used_user_agents

    # Jika sudah ada 100 User-Agent yang digunakan, reset daftar
    if len(used_user_agents) >= 100:
        used_user_agents = []

    # Pilih User-Agent yang belum digunakan dalam rotasi saat ini
    available_agents = [ua for ua in user_agents if ua not in used_user_agents]
    if not available_agents:
        available_agents = user_agents  # Jika semua sudah digunakan, reset ke seluruh User-Agent

    selected_agent = random.choice(available_agents)
    used_user_agents.append(selected_agent)
    return selected_agent

# Simulasi delay manusiawi
def human_like_delay():
    time.sleep(random.uniform(10, 45))  # Delay antara 10 hingga 45 detik

# Fungsi scraping untuk setiap URL
def scrape_url(url, session, max_retries=3):
    for attempt in range(max_retries):
        headers = {
            "User-Agent": get_random_user_agent(),
            "Referer": "https://www.google.com/",
            "Accept-Language": "en-US,en;q=0.9"
        }

        try:
            response = session.get(url, headers=headers, timeout=30)

            # Pastikan request berhasil
            if response.status_code == 429:
                print(f"Error 429: Terlalu banyak permintaan. Menunggu 10 menit sebelum mencoba lagi...")
                time.sleep(600)  # Jeda 10 menit untuk error 429
                continue
            elif response.status_code == 404:
                print(f"Error 404: URL tidak ditemukan: {url}. Melewati URL ini.")
                return None  # Skip URL
            elif response.status_code != 200:
                print(f"Gagal mengakses URL: {url}, status code: {response.status_code}")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue

            # Parse HTML menggunakan BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Mengambil harga rumah
            harga_elem = soup.select_one('div.flex.items-baseline.gap-x-1 span.text-primary.font-bold')
            harga = harga_elem.text.strip() if harga_elem else None

            # Mengambil lokasi
            lokasi_elem = soup.select_one('p.text-xs.text-gray-500.mb-2')
            lokasi = lokasi_elem.text.strip() if lokasi_elem else None

            # Mengambil spesifikasi
            spesifikasi = {}
            spesifikasi_elems = soup.find_all('div', class_=re.compile(r'flex.*gap-4.*'))
            if spesifikasi_elems:
                for elem in spesifikasi_elems:
                    key_elem = elem.select_one('p.w-32.text-xs.font-light.text-gray-500')
                    value_elem = elem.select_one('p:nth-of-type(2)')
                    if key_elem and value_elem:
                        key = key_elem.text.strip()
                        value = value_elem.text.strip()
                        spesifikasi[key] = value

            # Menambahkan informasi tambahan jika tidak tersedia
            if "Lainnya" not in spesifikasi:
                spesifikasi["Lainnya"] = "Informasi tambahan tidak tersedia"

            # Membuat dictionary hasil scraping
            data = {
                "URL": url,
                "Harga": harga,
                "Lokasi": lokasi,
                **spesifikasi
            }
            return data

        except Exception as e:
            print(f"Error saat scraping URL {url} pada percobaan ke-{attempt + 1}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    print(f"Gagal mengambil data setelah {max_retries} percobaan: {url}")
    return None

# Main scraping process
start_time = time.time()  # Mulai timer
all_data = []

try:
    with requests.Session() as session:
        for i, url in enumerate(tqdm(urls_to_scrape, desc="Scraping progress")):
            # Scrape URL
            data = scrape_url(url, session)
            if data:
                all_data.append(data)
            else:
                print(f"Error mengambil data di URL ke-{start_index + i}. Melewati URL ini.")

            # Delay manusiawi
            human_like_delay()

            # Jeda tambahan setiap 50 URL untuk menghindari deteksi bot
            if i > 0 and i % 50 == 0:
                print("Jeda tambahan 5 menit untuk menghindari deteksi bot...")
                time.sleep(300)  # Jeda 5 menit

    # Simpan hasil scraping ke CSV jika ada data yang berhasil diambil
    if all_data:
        # Jika file sudah ada, gabungkan data baru dengan data lama
        if os.path.exists(output_file):
            new_data = pd.DataFrame(all_data)
            combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=["URL"])
        else:
            combined_data = pd.DataFrame(all_data)

        # Simpan ke file
        combined_data.to_csv(output_file, index=False)
        print(f"Scraping selesai. {len(all_data)} data baru ditambahkan. Total {len(combined_data)} data disimpan dalam '{output_file}'.")
    else:
        print("Tidak ada data baru yang ditemukan.")
except Exception as e:
    print(f"Proses scraping dihentikan karena error: {e}")
    winsound.Beep(440, 1000)  # Bunyi notifikasi saat error

# Hitung durasi waktu scraping
end_time = time.time()
elapsed_time = end_time - start_time

# Format durasi
if elapsed_time < 3600:
    formatted_time = str(timedelta(seconds=int(elapsed_time)))  # Format MM:SS
else:
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  # Format HH:MM:SS

print(f"Durasi scraping: {formatted_time}")
winsound.Beep(660, 1000)  # Bunyi notifikasi saat selesai
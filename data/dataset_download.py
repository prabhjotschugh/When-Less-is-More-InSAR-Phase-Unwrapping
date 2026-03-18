import os
import json
import random
import time
import warnings
import requests
import numpy as np
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# 1. CONFIGURATION
class Config:
    BASE_DIR = "insar_icml_project"
    DATA_DIR = os.path.join(BASE_DIR, "raw_frames")
    SPLIT_FILE = os.path.join(BASE_DIR, "dataset_splits.json")
    MODEL_PATH = os.path.join(BASE_DIR, "insar_unet_icml.pth")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")



    FRAMES_TO_DOWNLOAD = {
        # Calatrava Volcanic Field (Spain)
        '001A_05031_131313' : {
            'interferograms': ['20210916_20210928', '20210922_20220309', '20211028_20211115', '20221222_20230115', '20220120_20220309', '20220225_20220426', '20220309_20230304', '20220414_20220613', '20220520_20220707', '20220613_20220905', '20220707_20220917', '20220812_20221011', '20220917_20230316', '20221116_20230103', '20221116_20221128', '20230127_20230316', '20230328_20230620', '20230608_20230912', '20230924_20231217', '20240310_20240614']
        },
        # Pico, San Jorge, Terceira (Azores)
        '002A_05136_020502' : {
            'interferograms': ['20230103_20230127', '20230220_20230328', '20230328_20230620', '20230527_20230714', '20230702_20230819', '20230831_20231018', '20231018_20231205', '20231123_20240110', '20231229_20240215', '20240122_20240310', '20240227_20240415', '20240415_20240521', '20240521_20240708', '20240626_20240813', '20240801_20240918', '20240930_20241117', '20241105_20241223', '20241223_20250209', '20250209_20250329', '20250504_20250528']
        },
        # Pico de Orizaba (Mexico)
        '005A_07021_131313' : {
            'interferograms': ['20210719_20210731', '20210818_20211210', '20210905_20220322', '20210917_20220626', '20211222_20220614', '20220310_20220930', '20220602_20230317', '20220825_20230128', '20220906_20230221', '20221117_20230104', '20230128_20230317', '20230317_20230913', '20230528_20230715', '20230901_20231019', '20231124_20240111', '20240216_20240404', '20240615_20240919', '20240907_20250330', '20241224_20250210', '20250610_20250704']
        },
        # Mayor Island (New Zealand)
        '008A_12731_060000' : {
            'interferograms': ['20250105_20250222', '20230528_20230715', '20240311_20250306', '20241025_20241212', '20250523_20250722', '20230609_20230925', '20240721_20240826', '20250210_20250505', '20230913_20231031', '20240111_20240311', '20250628_20250908', '20240510_20240627', '20231206_20240123', '20240204_20240416', '20250411_20250511', '20231112_20231230', '20240907_20241025', '20250827_20251002', '20241224_20250210', '20230808_20230925']
        },
        # Okataina, White Island (New Zealand)
        '008A_12836_151207' : {
            'interferograms': ['20210414_20210508', '20210601_20220602', '20210719_20211210', '20210929_20220626', '20211116_20220403', '20211222_20230116', '20220403_20220521', '20220602_20230305', '20220801_20230104', '20220930_20230317', '20221105_20221223', '20221223_20230128', '20230221_20230410', '20230422_20230609', '20230621_20230925', '20230808_20230901', '20231124_20240111', '20240204_20240627', '20240919_20241106', '20250105_20250411']
        },
        # Middle Gobi (Mongolia)
        '011A_04472_131313' : {
            'interferograms': ['20250505_20250716', '20200823_20200928', '20230715_20231206', '20210303_20220322', '20220403_20230715', '20201103_20210207', '20210701_20210818', '20230621_20240323', '20211204_20220121', '20200904_20210923', '20220202_20220322', '20210408_20210701', '20230422_20230715', '20201010_20201127', '20210502_20210818', '20220801_20230422', '20211029_20211204', '20200916_20201103', '20220226_20220403', '20210126_20210315']
        },
        # Ranakah, Sangeang Api, Wai Sano, poco_leok
        '010A_09915_111413' : {
            'interferograms': ['20220202_20220226', '20220602_20230317', '20230104_20230221', '20230504_20230621', '20240123_20240311', '20240814_20240907', '20250105_20250222', '20250505_20250809', '20220310_20220930', '20220825_20230128', '20230221_20230422', '20230703_20230820', '20240216_20240428', '20240919_20250318', '20250210_20250330', '20250728_20250914', '20220509_20220626', '20221129_20230104', '20230422_20230609', '20231206_20240123']
        },
        # Melbourne, Pleiades, The, Ritmann, Mount
        '010A_16318_111313' : {
            'interferograms': ['20221223_20230410', '20230116_20230221', '20230221_20230410', '20230317_20230504', '20230422_20230609', '20230504_20230516', '20230621_20230913', '20230727_20230901', '20230820_20231007', '20230913_20231031', '20231031_20231218', '20231124_20240111', '20231230_20240216', '20240123_20240311', '20240228_20240404', '20240323_20240416', '20240522_20240627', '20240709_20240802', '20240907_20250318', '20250411_20250505']
        },
        # Black Rock Desert2, Markagunt Plateau
        '020A_05163_131313' : {
            'interferograms': ['20200103_20200220', '20200315_20200502', '20200514_20200701', '20200713_20201216', '20200818_20210214', '20201216_20210202', '20210121_20210226', '20210226_20210403', '20210322_20210427', '20210415_20210521', '20210509_20210614', '20210602_20210708', '20210708_20210813', '20210813_20210918', '20210906_20211012', '20211024_20211129', '20250406_20250524', '20250605_20250921', '20250816_20251015', '20251027_20251202']
        },
        # San Francisco Volcanic Field, Uinkaret Field
        '020A_05362_131313' : {
            'interferograms': ['20200315_20200502', '20200420_20200607', '20200526_20200607', '20210427_20210509', '20210509_20210626', '20210602_20210720', '20210614_20210801', '20210626_20210906', '20210708_20210825', '20210720_20210906', '20210801_20210918', '20210825_20210930', '20210906_20211012', '20210930_20211105', '20211012_20211117', '20211105_20211211', '20211129_20211223', '20250406_20250524', '20250605_20250723', '20251003_20251202']
        },
        # Sabalan
        '006D_05111_131313' : {
            'interferograms': ['20200425_20200507','20200519_20200706','20200612_20210303','20200904_20210911','20201010_20201127','20201127_20210114','20201221_20210207','20210126_20210315','20210303_20210619','20210327_20210923','20210420_20210607','20210526_20210713','20210619_20210911','20210701_20210830','20210818_20211005','20210923_20211110','20211029_20211216','20211228_20220214','20220226_20220403','20231124_20240111']
        },
        # Sahand
        '006D_05310_131313' : {
            'interferograms': ['20210607_20220322','20210613_20210701','20210719_20210818','20210911_20220310','20211110_20211228','20211216_20220202','20220121_20220310','20220322_20220509','20220427_20220614','20220614_20220930','20220720_20220906','20220918_20221105','20221117_20230104','20221211_20230128','20230116_20230305','20230317_20230925','20230609_20230901','20230925_20231112','20231206_20240123','20240311_20240919']
        },
        # Kawi-Butak, Kelut, Lawu, Penanggungan, Semeru, Tengger Caldera, Wilis, malang_plain
        '003D_09757_111111' : {
            'interferograms': ['20230702_20230726','20230702_20240215','20230726_20240626','20240215_20240720','20240521_20240801','20240626_20250609','20240708_20240801','20250516_20250621','20250516_20250820','20250609_20250913','20250621_20250925','20250808_20250901','20250820_20251007','20250901_20251019','20250913_20251031','20250925_20251112','20251007_20251112','20251019_20251112','20250808_20250925','20240626_20240720']
        },
        # Methana
        '007D_05293_151310' : {
            'interferograms': ['20210818_20210905','20210917_20220310','20210923_20220918','20211011_20211104','20211128_20211222','20211210_20211228','20220121_20220310','20220310_20220626','20220322_20220930','20220614_20220918','20220813_20220930','20220918_20221105','20221117_20230104','20230116_20230305','20230317_20230925','20230609_20230925','20230913_20231112','20231206_20240123','20240311_20240919','20240603_20250306']
        },
        # Deception Island, Melville, Penguin Island
        '009D_15291_161402' : {
            'interferograms': ['20210725_20210806','20210725_20221012','20210806_20220509','20210911_20220310','20211005_20220708','20211110_20220521','20211228_20220214','20220121_20220310','20220310_20230305','20220322_20230317','20220509_20230808','20220602_20230901','20220614_20230305','20220708_20231007','20220801_20230504','20220906_20230317','20221012_20230727','20221105_20230820','20230305_20240311','20230609_20240627']
        },
        # Cerro Hudson, Macá, Meullín, Río Murta
        '010D_13610_131313' : {
            'interferograms': ['20200102_20200114','20200126_20200326','20200219_20200407','20200302_20200910','20200314_20210321','20200419_20200712','20200501_20200513','20200724_20200910','20200817_20201004','20200910_20210905','20200922_20210917','20201016_20201203','20201121_20210108','20201227_20210213','20210120_20210309','20210225_20210414','20210309_20210917','20210426_20210905','20210508_20210917','20211116_20211222']
        },
        # Aguilera, Lautaro, Reclus, Viedma
        '010D_13986_131310' : {
            'interferograms': ['20211116_20211128','20211116_20211210','20211116_20211222','20211128_20211210','20211128_20211222','20211210_20211222']
        },
        # Durango Volcanic Field
        '012D_06537_131313' : {
            'interferograms': ['20220427_20220930','20220906_20230621','20221012_20230116','20221211_20230128','20230116_20230305','20230317_20230516','20230422_20230621','20230516_20230820','20230621_20230925','20230808_20231007','20230901_20231019','20231031_20231218','20231206_20240123','20240111_20240228','20240216_20240404','20240311_20250318','20240416_20240826','20240907_20250306','20250105_20250222','20250902_20251020']
        },
        # Nunivak Island
        '015D_02942_110000' : {
            'interferograms': ['20200425_20200531','20200519_20200706','20200612_20210619','20200718_20200916','20200823_20201010','20200928_20201115','20201022_20201127','20201115_20210514','20201127_20210514','20210514_20210701','20210526_20210713','20210607_20210725','20210701_20210806','20200624_20200823','20200706_20200904','20200811_20200928','20200904_20201022','20201010_20201127','20201127_20210526','20210502_20210619']
        },
        # Garibaldi Lake
        '086D_04090_131308' : {
            'interferograms': ['20200107_20200224', '20200212_20200412', '20200331_20200518', '20200424_20200611', '20200518_20200717', '20200611_20200927', '20200729_20200915', '20200822_20201009', '20200915_20201102', '20201021_20201208', '20201114_20210101', '20201208_20210125', '20210125_20210314', '20210218_20210407', '20210302_20210922', '20210419_20210606', '20210525_20210718', '20210618_20210910', '20210730_20210910', '20211016_20211203']
        }
    }


    BASE_URL_TEMPLATES = [
        "https://data.ceda.ac.uk/neodc/comet/data/licsar_products/{track}",
        "https://dap.ceda.ac.uk/neodc/comet/data/licsar_products/{track}",
        "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products.public/{track}",
        "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/{track}"
    ]
    
    PATCH_SIZE = 128
    STRIDE = 64
    MIN_COHERENCE = 0.5  
    MIN_LOS_MAGNITUDE = 0.1
    BATCH_SIZE = 32
    EPOCHS = 500
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    WAVELENGTH = 0.056
    IN_CHANNELS = 6
    OUT_CHANNELS = 1

cfg = Config()
for d in [cfg.BASE_DIR, cfg.DATA_DIR, cfg.RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 2. ROBUST DATA DOWNLOADER
def extract_track_number(frame_id):
    """Extract track number and ensure correct folder padding (e.g., '2' -> '02')."""
    track_with_direction = frame_id.split('_')[0]
    track_number = track_with_direction[:-1].lstrip('0')
    if not track_number: track_number = '0'
    # LiCSAR servers use 2-digit padding for tracks < 100 (e.g. 02, 10, 150)
    return track_number.zfill(2)

def download_file_with_fallback(base_urls, output_path, path_variants, max_retries=3):
    """Try multiple URLs with built-in retries for Timeouts and Connection Errors."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    # Added headers to simulate a browser; academic servers often throttle default python-requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for base_url in base_urls:
        for path_variant in path_variants:
            url = f"{base_url}/{path_variant}"
            
            for attempt in range(max_retries):
                try:
                    # Increased timeout to 100s for slow JASMIN public links
                    response = requests.get(url, stream=True, timeout=150, headers=headers)
                    
                    if response.status_code == 404: 
                        break # Try next variant or next base_url
                    
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    with open(output_path, 'wb') as f, tqdm(
                        desc=f"  {os.path.basename(output_path)}",
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=32768): # Larger chunk size for speed
                            if chunk:
                                size = f.write(chunk)
                                pbar.update(size)
                    return True

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        print(f"  ⚠ {type(e).__name__}, retrying ({attempt+1}/{max_retries})...")
                        time.sleep(5 * (attempt + 1)) # Incremental backoff
                        continue
                except Exception:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    break 
    
    print(f"  ✗ FAILED to download {os.path.basename(output_path)} after all attempts")
    return False

def run_downloader():
    print(f"\n{'='*60}\nSTEP 1: DOWNLOADING DATA ({len(cfg.FRAMES_TO_DOWNLOAD)} FRAMES)\n{'='*60}")
    
    files_to_dl = ['geo.diff_pha.tif', 'geo.unw.tif', 'geo.cc.tif']
    meta_to_dl = ['geo.E.tif', 'geo.N.tif', 'geo.U.tif']
    
    for frame_id, info in cfg.FRAMES_TO_DOWNLOAD.items():
        # IMPROVEMENT: We now generate BOTH padded (05) and unpadded (5) track versions
        track_padded = extract_track_number(frame_id)
        track_unpadded = track_padded.lstrip('0') if track_padded != "00" else "0"
        
        frame_dir = os.path.join(cfg.DATA_DIR, frame_id)
        os.makedirs(frame_dir, exist_ok=True)
        
        # Build base_urls list using BOTH versions of the track number
        base_urls = []
        for track_ver in [track_padded, track_unpadded]:
            for t in cfg.BASE_URL_TEMPLATES:
                base_urls.append(t.format(track=track_ver))
        
        # 1. Metadata
        meta_dir = os.path.join(frame_dir, 'metadata')
        os.makedirs(meta_dir, exist_ok=True)
        for meta_file in meta_to_dl:
            out_path = os.path.join(meta_dir, f"{frame_id}.{meta_file}")
            if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
                print(f"Downloading {meta_file} for {frame_id}...")
                variants = [f"{frame_id}/metadata/{frame_id}.{meta_file}"]
                download_file_with_fallback(base_urls, out_path, variants)

        # 2. Interferograms
        ifg_root = os.path.join(frame_dir, 'interferograms')
        os.makedirs(ifg_root, exist_ok=True)
        
        for ifg_id in info['interferograms']:
            print(f"📡 Processing interferogram: {ifg_id}")
            ifg_dir = os.path.join(ifg_root, ifg_id)
            os.makedirs(ifg_dir, exist_ok=True)
            
            for f_name in files_to_dl:
                full_name = f"{ifg_id}.{f_name}"
                out_path = os.path.join(ifg_dir, full_name)
                
                if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
                    print(f"  Downloading {f_name}...")
                    variants = [
                        f"{frame_id}/{ifg_id}/{full_name}",
                        f"{frame_id}/interferograms/{ifg_id}/{full_name}"
                    ]
                    download_file_with_fallback(base_urls, out_path, variants)

    print("\n✓ Download Process Finished.")

run_downloader()
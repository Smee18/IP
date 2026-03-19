import h5py
import numpy as np
from tqdm import tqdm
from sky import sigma_clip
from photutils.psf.matching import TukeyWindow
from PIL import Image

raw_path = 'data/Galaxy10_DECals.h5'
processed_path = 'data/Galaxy10_ProcessedandCroppedV4.h5'

Q = 2
alpha = 0.02
taper_2d = TukeyWindow(alpha=0.1)((256, 256))

with h5py.File(raw_path, 'r') as F_raw, h5py.File(processed_path, 'w') as F_proc:
    n_images = F_raw['images'].shape[0]
    
    # Create dynamically resizable datasets to handle dropped NaN images
    dset_img = F_proc.create_dataset('images', shape=(0, 128, 128), maxshape=(None, 128, 128), dtype='float32', compression='gzip')
    dset_ans = F_proc.create_dataset('ans', shape=(0,), maxshape=(None,), dtype='uint8')

    discarded = 0

    batch_size = 500
    for start in tqdm(range(0, n_images, batch_size), desc="Extracting & Scaling Galaxies"):
        end = min(start + batch_size, n_images)
        
        batch_raw = np.array(F_raw['images'][start:end])
        batch_redshift = np.array(F_raw['redshift'][start:end])
        batch_ans = np.array(F_raw['ans'][start:end])
        
        valid_processed = []
        valid_ans = []
        
        for i in range(end - start):
            z = batch_redshift[i]
            
            # The Data Gateway: Drop NaNs, negative, or zero redshifts instantly
            if np.isnan(z) or z < 0.001:
                discarded += 1
                continue
            
            # 1. Target Scale Calculation
            nz_int = int(7.35 / z)
            half_width = nz_int // 2

            if nz_int < 64 or nz_int > 240:
                discarded += 1
                continue
            
            raw_flux = batch_raw[i, ..., 1].astype(np.float32) / 255.0
            
            # 2. Sky Subtraction
            mask = sigma_clip(raw_flux, alpha=3, verbose=False)
            sky_level = np.nanmedian(raw_flux[mask])
            sky_std = np.nanstd(raw_flux[mask])
            
            clean_flux = np.maximum(raw_flux - sky_level, 0)
            #lean_flux[clean_flux < (2 * sky_std)] = 0 

            # 3. Clean Bounding Box Constraints
            x_min = max(0, 128 - half_width)
            x_max = min(256, 128 + half_width)
            y_min = max(0, 128 - half_width)
            y_max = min(256, 128 + half_width)

            # 4. Geometric Transformation (Warping)
            if nz_int > 256:
                # Padding condition for massive physical footprints
                temp_storage = np.zeros((nz_int, nz_int), dtype='float32')
                pad = (nz_int - 256) // 2
                temp_storage[pad:pad+256, pad:pad+256] = clean_flux
                pil_img = Image.fromarray(temp_storage)
            else:
                # Cropping condition for smaller physical footprints
                cropped_image = clean_flux[y_min:y_max, x_min:x_max]
                pil_img = Image.fromarray(cropped_image)
            
            # High-fidelity interpolation back to standard network size
            upscaled_image = np.array(pil_img.resize((256, 256), Image.LANCZOS))
            
            # 5. Taper and Stretch
            tapered = upscaled_image * taper_2d
            final_crop = tapered[64:192, 64:192]
            final_crop = final_crop / (np.max(final_crop) + 1e-9)
            final_image = np.arcsinh(alpha * Q *final_crop) / Q
            
            valid_processed.append(final_image) ## CAREFUL THIS IS TAPERED
            valid_ans.append(batch_ans[i])
            
        # 6. Dynamic Disk Write
        if valid_processed:
            current_len = dset_img.shape[0]
            add_len = len(valid_processed)
            
            dset_img.resize(current_len + add_len, axis=0)
            dset_ans.resize(current_len + add_len, axis=0)
            
            dset_img[current_len:] = np.stack(valid_processed)
            dset_ans[current_len:] = np.array(valid_ans)

print(f"Removed {discarded} samples")
print("Preprocessing complete. Invalid samples dropped. Output dataset aligned.")
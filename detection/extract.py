import os
import tempfile
from pathlib import Path
import requests
import numpy as np
import fitz
from PIL import Image, ImageDraw
import cv2

# Dilation kernel size (in pixels) - increase this to expand the mask more
DILATION_KERNEL_SIZE = 10  # 50x50 kernel

def add_white_background_to_rectangle(image_with_alpha, padding_percentage=10):
    """
    Add white background around a cropped image to make it rectangular.
    The function adds padding based on the image's dimensions.
    
    Args:
        image_with_alpha: PIL Image with alpha channel (RGBA)
        padding_percentage: Percentage of the largest dimension to add as padding (default 10%)
    
    Returns:
        PIL Image (RGB) with white background and padding
    """
    # Get the bounding box of non-transparent pixels
    bbox = image_with_alpha.getbbox()
    
    if bbox is None:
        # If image is completely transparent, return white background
        white_bg = Image.new('RGB', image_with_alpha.size, 'white')
        return white_bg
    
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate the dimensions of the non-transparent content
    content_width = x_max - x_min
    content_height = y_max - y_min
    
    # Calculate padding based on the largest dimension
    max_dimension = max(content_width, content_height)
    padding = int(max_dimension * (padding_percentage / 100))
    
    # Create new dimensions with padding
    new_width = content_width + (padding * 2)
    new_height = content_height + (padding * 2)
    
    # Create white background
    result_image = Image.new('RGB', (new_width, new_height), 'white')
    
    # Paste the original image on top of white background
    # Convert RGBA to RGB for pasting with transparency
    rgb_image = image_with_alpha.convert('RGB')
    
    # Paste with alpha mask to handle transparency properly
    result_image.paste(rgb_image, (padding, padding), image_with_alpha)
    
    return result_image


def extract_and_crop_by_mask(original_image, images):
    """
    Extract segmentation masks from API response and crop image based on mask.
    Only the masked pixels are kept. Mask is dilated to include edges.
    Adds white background to make the image rectangular.
    
    Args:
        response_json: The JSON response from the API
        filename: Original filename
        page_num: Page number (for PDFs), None for single images
        image_path: Path to the original image for reference
    """
    try:
        for _, img_data in enumerate(images):
            image_shape = img_data.get('shape', [])
            results = img_data.get('results', [])
            
            if not image_shape or not results:
                print(f"  No results found in response")
                continue
            
            height, width = image_shape[0], image_shape[1]
            print(f"  Image shape: {width}x{height}")
            
            # Process each detected object
            for obj_idx, result in enumerate(results):
                name = result.get('name', 'Unknown')
                confidence = result.get('confidence', 0)
                segments = result.get('segments', {})
                
                if not segments or 'x' not in segments or 'y' not in segments:
                    print(f"    Object {obj_idx} ({name}): No segments found")
                    continue
                
                seg_x = segments['x']
                seg_y = segments['y']
                
                print(f"    Confidence: {confidence:.2%})")
                
                # Create mask from polygon
                mask = create_mask_from_polygon(width, height, seg_x, seg_y)
                
                # Dilate the mask to include edges
                dilated_mask = dilate_mask(mask, kernel_size=DILATION_KERNEL_SIZE)
                
                # Get the bounding box of the dilated mask (non-zero pixels)
                mask_bbox = dilated_mask.getbbox()
                
                if mask_bbox is None:
                    print(f"      Mask is empty, skipping")
                    continue
                
                x_min, y_min, x_max, y_max = mask_bbox
                print(f"      Mask bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                print(f"      Mask dilated with kernel size: {DILATION_KERNEL_SIZE}")
                
                # Crop both image and mask to the bounding box
                cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
                cropped_mask = dilated_mask.crop((x_min, y_min, x_max, y_max))

                # Create RGBA image with alpha channel from mask
                cropped_rgba = Image.new('RGBA', cropped_image.size)
                cropped_rgba.paste(cropped_image, (0, 0))
                
                # Set alpha channel based on dilated mask
                alpha_channel = cropped_mask.convert('L')
                cropped_rgba.putalpha(alpha_channel)

                # Add white background to make rectangular
                final_image = add_white_background_to_rectangle(cropped_rgba, padding_percentage=10)
                
                return final_image
    
    except Exception as e:
        print(f"  Error extracting masks: {e}")


def create_mask_from_polygon(width, height, x_coords, y_coords):
    """
    Create a binary mask image from polygon coordinates.
    
    Args:
        width: Image width
        height: Image height
        x_coords: List of x coordinates
        y_coords: List of y coordinates
    
    Returns:
        PIL Image with the binary mask
    """
    # Create a black image
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Create polygon points
    points = list(zip(x_coords, y_coords))
    
    # Draw filled polygon
    draw.polygon(points, fill=255)
    
    return mask


def dilate_mask(mask, kernel_size=5):
    """
    Dilate the mask to expand it slightly and ensure edges are included.
    
    Args:
        mask: PIL Image mask (binary, 0 or 255)
        kernel_size: Size of the dilation kernel (odd number like 3, 5, 7, etc.)
    
    Returns:
        Dilated PIL Image mask
    """
    # Convert PIL image to numpy array
    mask_array = np.array(mask)
    
    # Create dilation kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply dilation
    dilated_array = cv2.dilate(mask_array, kernel, iterations=1)
    
    # Convert back to PIL Image
    dilated_mask = Image.fromarray(dilated_array, mode='L')
    
    return dilated_mask


if __name__ == "__main__":
    # List all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip if it's a directory
        if not os.path.isfile(filepath):
            continue
        
        print(f"Processing: {filename}")
        
        # Check if it's a PDF
        if filename.lower().endswith('.pdf'):
            try:
                # Convert PDF to PNG images
                images = fitz.open(filepath)
                
                # Process each page
                for i, image in enumerate(images):
                    # Store PNG to tempfile

                    pix = image.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                        image.save(tmp_path, 'PNG')
                        print(f"  Temporary PNG created: {tmp_path}")
                    
                    try:
                        # Send HTTP request with the temporary PNG
                        with open(tmp_path, 'rb') as f:
                            response = requests.post(
                                url,
                                headers={"Authorization": f"Bearer {api_key}"},
                                data=args,
                                files={"file": f},
                            )
                        
                        print(f"  Response status: {response.status_code}")
                        
                        # Extract and crop based on mask
                        extract_and_crop_by_mask(response.json(), filename, i, tmp_path)
                    
                    except requests.exceptions.RequestException as e:
                        print(f"  Error sending request: {e}")
                    
                    finally:
                        # Clean up tempfile
                        try:
                            os.unlink(tmp_path)
                            print(f"  Temporary file deleted: {tmp_path}")
                        except Exception as e:
                            print(f"  Error deleting temp file: {e}")
            
            except Exception as e:
                print(f"  Error processing PDF {filename}: {e}")
        
        # Check if it's a JPG
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            try:
                # Send HTTP request with the JPG
                with open(filepath, 'rb') as f:
                    response = requests.post(
                        url,
                        headers={"Authorization": f"Bearer {api_key}"},
                        data=args,
                        files={"file": f},
                    )
                
                print(f"  Response status: {response.status_code}")
                
                # Extract and crop based on mask
                extract_and_crop_by_mask(response.json(), filename, None, filepath)
            
            except requests.exceptions.RequestException as e:
                print(f"  Error sending request: {e}")
        
        else:
            print(f"  Skipped: Not a PDF or JPG file")

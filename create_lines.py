import os
import time
import textwrap
import numpy as np
import pandas as pd
import cv2
import random
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt

VALID_DATA_CSV_PATH = './valid_data.csv'
INVALID_DATA_CSV_PATH = './invalid_data.csv'
IMAGE_PATH = './data/lines'
IMG_WIDTH = 1600
IMG_HEIGHT = 128

## available fonts, their sizes and letter spacings ##
fonts_and_attrs = [
    {'font': 'IndieFlower-Regular.ttf',
     'font_sizes': [42, 44, 46, 48],
     'letter_spacings': [ -2, -1, 0, 1, 2],
     'text_wrap_plus': [20, 20, 20, 10]
    },

    {'font': 'Zeyada-Regular.ttf',
     'font_sizes': [44, 46, 48, 50, 52,],
     'letter_spacings': [-2, -1, 0, 1, 2],
     'text_wrap_plus': [20, 20, 10, 10, 10]
    },

    {'font': 'PatrickHand-Regular.ttf',
     'font_sizes': [42, 44, 46, 48, 50, 52,],
     'letter_spacings': [-3, -1, 0, 1, 2],
     'text_wrap_plus': [30, 30, 30, 20, 10, 10, 10]
    },

    {'font': 'Caveat-Regular.ttf',
     'font_sizes': [42, 44, 46, 48,],
     'letter_spacings': [ -1, 0],
     'text_wrap_plus': [10, 10, 10, 10]
    },

    {'font': 'CaveatBrush-Regular.ttf',
     'font_sizes': [42, 44, 46, 48, 50],
     'letter_spacings': [ -2, -1, 0, 1, 2],
     'text_wrap_plus': [20, 20, 20, 10, 10]
    },

    {'font': 'ShadowsIntoLight-Regular.ttf',
     'font_sizes': [42, 44, 46, 48, 50],
     'letter_spacings': [ -2, -1, 0, 1, 2],
     'text_wrap_plus': [ 20, 20, 20, 20, 10]
    },

    {'font': 'AnnieUseYourTelescope-Regular.ttf',
     'font_sizes': [42, 44, 46, 48, 50],
     'letter_spacings': [ -2, -1, 0, 1, 2],
     'text_wrap_plus': [ 20, 20, 20, 20, 20]
    },

    {'font': 'JustMeAgainDownHere-Regular.ttf',
     'font_sizes': [42, 44, 46, 48, 50],
     'letter_spacings': [-1, 0, 1, 2, 3],
     'text_wrap_plus': [ 30, 20, 20, 20, 20]
    },

    {'font': 'JustAnotherHand-Regular.ttf',
     'font_sizes': [44, 46, 48, 50, 52],
     'letter_spacings': [-1, 0, 1, 2, 3],
     'text_wrap_plus': [20, 20, 20, 20, 20]
    },

    {'font': 'GloriaHallelujah-Regular.ttf',
     'font_sizes': [42, 44, 46,],
     'letter_spacings': [-1, 0, 1],
     'text_wrap_plus': [ 10, 0, 0]
    },
]

def write_text(draw: ImageDraw.Draw, text:str, font: ImageFont, font_str: str, img_width: int, img_height: int, spacing: int):
    return_val = True
    total_text_width, total_text_height = draw.textsize(text, font=font, language = 'tr')
    width_difference = (img_width - 20) - total_text_width
    gap_width = int(width_difference / (total_text_width - 1))

    xpos = (img_width // 2) - (total_text_width // 2) - (0 if spacing < 0 else spacing * (len(text) // 2))
    ypos = (img_height // 2) - (total_text_height // 2)
    if font_str == 'Caveat-Regular.ttf':
        xpos -= 100

    if xpos < 0:
        return_val = False

    for letter in text:
        draw.text((xpos, ypos),letter, font = font, fill=(0),)
        letter_width, letter_height = draw.textsize(letter, font=font)
        xpos += letter_width + gap_width + spacing

    if xpos > img_width:
        return_val = False

    return return_val

VALID_LINE_COUNTER = 0
INVALID_LINE_COUNTER = 0
def create_lines_from_text(wrapped_text: list, file_indx: int):
    global VALID_LINE_COUNTER
    global INVALID_LINE_COUNTER

    font = ImageFont.truetype(os.path.join('fonts', font_str), font_size, encoding='utf-8')
    for i, text in enumerate(wrapped_text):
        img = np.ones((128, 1600), dtype = 'uint8') * 255
        img_pil = Image.fromarray(img)

        draw = ImageDraw.Draw(img_pil)
        x, y = draw.textsize(text, font, language='tr')

        # False if text overflows the image boundaries
        is_valid_line = True
        if x > IMG_WIDTH or y > IMG_HEIGHT:
            is_valid_line = False
        else:
            if not write_text(draw, text, font, font_str, IMG_WIDTH, IMG_HEIGHT, letter_space):
                is_valid_line = False    
            else:
                img = np.array(img_pil)
                img_path = os.path.join(IMAGE_PATH, '{}_{}.png'.format(file_indx, VALID_LINE_COUNTER))

                cv2.imwrite(img_path, img)
                #cv2.imshow(str(letter_space), img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

        if is_valid_line:
            valid_img_paths.append(img_path)
            # replace ; with ## to avoid csv shifting text to next column
            valid_transcribes.append(str(text).replace(';', '##'))
            valid_font_sizes.append(font_size)
            valid_letter_spaces.append(letter_space)
            valid_font_types.append(font_str.split('.')[0])
            VALID_LINE_COUNTER += 1
        else:
            invalid_transcribes.append(str(text).replace(';', '##'))
            invalid_font_sizes.append(font_size)
            invalid_letter_spaces.append(letter_space)
            invalid_font_types.append(font_str.split('.')[0])
            INVALID_LINE_COUNTER += 1

# for reproducability
random.seed(3)

df = pd.read_csv('texts.csv')
valid_img_paths   = []
valid_transcribes = []
valid_font_sizes  = []
valid_letter_spaces = []
valid_font_types = []

invalid_transcribes = []
invalid_font_sizes  = []
invalid_letter_spaces = []
invalid_font_types = []

def get_random_font_and_attributes():
    attr_dict = random.choice(fonts_and_attrs)
    font_str = attr_dict['font']
    const_font_sizes = attr_dict['font_sizes']
    const_letter_spacings = attr_dict['letter_spacings']
    text_wrap_values = attr_dict['text_wrap_plus']

    font_size = random.choice(const_font_sizes)
    letter_space = random.choice(const_letter_spacings)

    font_size_indx = list(const_font_sizes).index(font_size)
    text_wrap_value = text_wrap_values[font_size_indx]

    return font_str, font_size, letter_space, text_wrap_value

start_time = time.time()
for i in range(len(df)):
    sample = df.iloc[i]
    sample_text = str(sample.text)
    sample_text = ' '.join(sample_text.split())
    sample_text = sample_text.replace('`', '\'')
    sample_tokens = sample_text.split(' ')


    first_half_of_tokens = sample_tokens[: int(len(sample_tokens) // 2)]
    rest_of_tokens = sample_tokens[int(len(sample_tokens) // 2):]

    first_half_of_text = ' '.join(first_half_of_tokens)
    rest_of_text = ' '.join(rest_of_tokens)

    # get fonts and its attributes randomly for the first half of the text
    font_str, font_size, letter_space, text_wrap_value = get_random_font_and_attributes()
    wrapped_text = textwrap.wrap(first_half_of_text, width=font_size + text_wrap_value)
    create_lines_from_text(wrapped_text, i)

    # get fonts and its attributes randomly for the second half of the text
    font_str, font_size, letter_space, text_wrap_value = get_random_font_and_attributes()
    wrapped_text = textwrap.wrap(rest_of_text, width=font_size + text_wrap_value)
    create_lines_from_text(wrapped_text, i)

    print("Text #{} completed".format(i + 1))
end_time = time.time()

print('Elapsed time: ', end_time - start_time)
valid_df = pd.DataFrame({'image_path': valid_img_paths, 'transcribe': valid_transcribes, 
                        'font_size': valid_font_sizes, 'letter_space': valid_letter_spaces, 'font_type': valid_font_types})

invalid_df = pd.DataFrame({'transcribe': invalid_transcribes, 'font_size': invalid_font_sizes, 
                            'letter_space': invalid_letter_spaces, 'font_type': invalid_font_types})

valid_df.to_csv(VALID_DATA_CSV_PATH, index = False)
invalid_df.to_csv(INVALID_DATA_CSV_PATH, index = False)

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(api_key='AIzaSyCueIIN_zHmTwuuhLd5MTVCbattwLP6bgk')

response = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='Fuzzy bunnies in my kitchen',
    config=types.GenerateImagesConfig(
        number_of_images = 1,
    )
)
for generated_image in response.generated_images:
  image = Image.open(BytesIO(generated_image.image.image_bytes))
  image.show()

# response = client.models.generate_images(
#     model='imagen-3.0-generate-002',
#     prompt='부엌에 있는 귀여운 토끼들',
#     config=types.GenerateImagesConfig(
#         number_of_images=4,
#         output_mime_type='image/jpeg'
#     )
# )
#
# for generated_image in response.generated_images:
#     image = Image.open(BytesIO(generated_image.image.image_bytes))
#     image.show()
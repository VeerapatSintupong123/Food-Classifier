import base64

image_path = "image/massamun.jpg"
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

with open("encoded_image.txt", "w") as text_file:
    text_file.write(encoded_image)

print("Base64 image encoded and saved to 'encoded_image.txt'.")

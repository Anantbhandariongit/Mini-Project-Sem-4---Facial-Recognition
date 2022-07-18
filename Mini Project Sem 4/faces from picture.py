import face_recognition as fr
from PIL import Image, ImageDraw, ImageFont

anant_img = fr.load_image_file("Anant Bhandari.jpg")
anant_face_encoding = fr.face_encodings(anant_img)[0]

mom_img = fr.load_image_file("mataji.jpg")
mom_face_encoding = fr.face_encodings(mom_img)[0]

known_face_encodings = [
        anant_face_encoding, mom_face_encoding
     ]

known_face_names = [
    "Anant", "Upma"
]

img = fr.load_image_file("fam.jpg")

face_loc = fr.face_locations(img)
face_encoding = fr.face_encodings(img, face_loc)



pil_img = Image.fromarray(img)
draw = ImageDraw.Draw(pil_img)
font = ImageFont.truetype("arial.ttf", 25)

for(t, r, b, l), face_encoding in zip(face_loc, face_encoding):

    matches = fr.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    if True in matches:
        first_match = matches.index(True)

        name = known_face_names[first_match]
        
    text_width, text_height = draw.textsize(name)
     
    draw.rectangle(((l,b - text_height - 10), (r,b)), fill = (0, 0, 255), outline = (0, 0, 255))
    draw.text((l + 6, b - text_height - 5), name,font = font, fill = (255, 255, 255, 255))

del draw
pil_img.show()

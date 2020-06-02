from openpyxl import Workbook

book = Workbook()
sheet = book.active
sheet['A1'] = "frameNo"

frcnt = 0

for i in range(10):
    sheet['A' + str(frcnt + 3)] = frcnt
    frcnt += 1

faces = [
    (1088, 379, 1142, 433),
    (474, 380, 531, 437),
    (1606, 402, 1654, 450),
    (1270, 399, 1346, 475),
    (1004, 404, 1080, 480),
    (316, 428, 393, 505),
    (216, 450, 291, 525),
    (1161, 446, 1229, 514),
    (100, 477, 169, 546)
]

columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
           'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM'
           ]
clm = 1

face_to_search = "(1161, 446, 1229, 514)"

for face in faces:
    sheet[columns[clm] + str(1)] = str(face)
    sheet[columns[clm] + str(2)] = "face"+str(clm-1)
    clm += 1

face_search = "face8"

for fs in range(len(faces)):
    print(sheet[columns[fs + 1] + str(2)].value)
    if (sheet[columns[fs + 1] + str(2)].value == face_search):
        print("true")
        sheet[columns[fs + 1] + str(3)] = "emotion"

book.save("results.xlsx")

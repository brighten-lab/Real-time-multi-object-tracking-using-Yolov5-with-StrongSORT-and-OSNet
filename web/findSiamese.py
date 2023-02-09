from siamese.src import inference

class find:
    def similar(face):
        person = []
        path = "static/save_img/"
        for img_path in face:
            who = inference.run(path + img_path)
            person.append(who)
        return person
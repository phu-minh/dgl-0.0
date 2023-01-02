from deepface import DeepFace

def createModel(name='SFace'):
    model = DeepFace.build_model(name)
    return model


model = createModel('SFace')
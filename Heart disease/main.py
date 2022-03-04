from service.heartDiseaseService import HeartDiseaseService

if __name__ == '__main__':

    service = HeartDiseaseService()
    service.run_model()
    input("Press any key to exit ")

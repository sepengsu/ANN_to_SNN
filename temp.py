from model import vae as vqa

def main():
    model = vqa.Quant_VAE(3, 128)
    print(model.encoder)
    print(model.fc_mu)
    print(model.fc_var)
    print(model.decoder)



if __name__ == '__main__':
    main()
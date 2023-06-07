import gdown
import os



def download_weights(weights_name):
    destination = os.path.join(os.path.dirname(__file__), 'data', weights_name)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if weights_name == 'ViT-Base_Text_Emb_Hockey_Fighter.ckpt':
        # you can find the file id by opening the file in the browser in google drive and copying the id from the url
        file_id = '138GdG9GOlteVXU8s-1GC9sG9qy5HSOdK'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

        text_embeddings_file_id = '1qRcI4XxGwjshiT8IOPf1dUP90JUHCsbE'
        text_embeddings_url = f'https://drive.google.com/uc?id={text_embeddings_file_id}'
        text_embeddings_destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_object_label_embeddings.pt')
        gdown.download(text_embeddings_url, text_embeddings_destination, quiet=False)
    elif weights_name == 'ViT-Large_Text_Emb_Light_Sun.ckpt':
        file_id = '1wvNWKyPIM2Bldup_seNYLtdBbKT7vwTB'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

        text_embeddings_file_id = '1qRcI4XxGwjshiT8IOPf1dUP90JUHCsbE'
        text_embeddings_url = f'https://drive.google.com/uc?id={text_embeddings_file_id}'
        text_embeddings_destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_object_label_embeddings.pt')
        gdown.download(text_embeddings_url, text_embeddings_destination, quiet=False)
    elif weights_name == 'GraphCLIP.ckpt':
        # raise Exception()
        file_id = '1vhz_RRmdVhqSJ-ZC1y65-Lksd_zG-XM6'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

    elif weights_name == 'ViT-Large_Text_Emb_Spring_River.ckpt':
        file_id = '1a_m20zT5GjilVX7e_6tp8BtVzL9E2_Hc'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

        text_embeddings_file_id = '1qRcI4XxGwjshiT8IOPf1dUP90JUHCsbE'
        text_embeddings_url = f'https://drive.google.com/uc?id={text_embeddings_file_id}'
        text_embeddings_destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_object_label_embeddings.pt')
        gdown.download(text_embeddings_url, text_embeddings_destination, quiet=False)
        
        


def download_filtered_graphs():
    destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_graphs.pt')
    if not os.path.exists(destination):
        print('Downloading filtered graphs.')
        filed_id = '1xdQAjpATSAU-TR89CrqWp6aWXRctBDyz'
        url = f'https://drive.google.com/uc?id={filed_id}'
        gdown.download(url, destination, quiet=False)
    
    destination_test = os.path.join(os.path.dirname(__file__), 'data', 'filtered_graphs_test_small.pt')
    if not os.path.exists(destination_test):
        test_file_id = '1J2LQhXq8nvF4Bb1BW-I0zMmxdlANej7u'
        url = f'https://drive.google.com/uc?id={test_file_id}'
        gdown.download(url, destination_test, quiet=False)
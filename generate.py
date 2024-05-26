import torch
from model import CharRNN, CharLSTM
import numpy as np
import datetime
import random
import string
import torch.nn.functional as F  # 소프트맥스 함수를 사용하기 위해 추가

def generate_with_temperature(model, seed_characters, temperature, device, char_to_idx, idx_to_char, length=200):
    """ Generate characters with temperature parameter

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        device: device for computing, cpu or gpu
        char_to_idx: dictionary mapping characters to indices
        idx_to_char: dictionary mapping indices to characters
        length: number of characters to generate

    Returns:
        samples: generated characters
    """
    model.eval()  # 모델을 평가 모드로 설정
    hidden = model.init_hidden(1).to(device)  # 초기 은닉 상태를 GPU로 이동
    input_seq = torch.tensor([char_to_idx[c] for c in seed_characters]).unsqueeze(0).to(device)  # 시드 문자를 인덱스로 변환하여 텐서로 만들고 GPU로 이동

    samples = seed_characters  # 생성된 문자를 저장할 문자열 초기화
    for _ in range(length):
        output, hidden = model(input_seq, hidden)  # 모델을 통해 출력과 은닉 상태를 계산
        output = output.squeeze().div(temperature).exp()  # 온도 매개변수를 적용하여 출력 분포 조정
        probabilities = F.softmax(output, dim=-1).cpu()  # 소프트맥스 함수를 적용하여 확률 분포 계산
        top_i = torch.multinomial(probabilities, 1)[-1]  # 확률 분포를 사용하여 다음 문자의 인덱스를 샘플링
        char = idx_to_char[top_i.item()]  # 샘플링된 인덱스를 문자로 변환
        samples += char  # 생성된 문자를 결과 문자열에 추가
        input_seq = torch.tensor([[top_i.item()]]).to(device)  # 다음 입력 시퀀스를 업데이트하여 GPU로 이동

    return samples

def main():
    # 하이퍼파라미터 및 모델 구성
    hidden_size = 128
    num_layers = 2
    model_type = 'RNN'  # LSTM 모델을 사용하려면 'LSTM'으로 변경
    temperature = 0.8

    # A에서 Z 사이의 랜덤 시드 문자 생성
    seed_characters_list = [''.join(random.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(5)]
    
    # 데이터셋 로드하여 char_to_idx와 idx_to_char 가져오기
    from dataset import Shakespeare
    dataset = Shakespeare('shakespeare_train.txt')
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char

    # 최적의 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('best_rnn_model.pth' if model_type == 'RNN' else 'best_lstm_model.pth')
    vocab_size = checkpoint['vocab_size']
    output_size = vocab_size
    if model_type == 'RNN':
        model = CharRNN(vocab_size, hidden_size, output_size, num_layers).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = CharLSTM(vocab_size, hidden_size, output_size, num_layers).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 텍스트 샘플 생성
    length = 200
    current_time = datetime.datetime.now().strftime("%H%M%S")
    filename = f"generate_{current_time}.txt"
    with open(filename, 'w') as f:
        for seed_characters in seed_characters_list:
            generated_text = generate_with_temperature(model, seed_characters, temperature, device, char_to_idx, idx_to_char, length)
            f.write(f"Seed characters: {seed_characters}\n")
            f.write(f"Generated text: {generated_text}\n\n")
            print(f"Seed characters: {seed_characters}")
            print(f"Generated text: {generated_text}\n")

if __name__ == '__main__':
    main()

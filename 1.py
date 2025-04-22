import zipfile
import os

def unzip_special_filename(zip_file_path, extract_path):
    """
    특수 문자가 포함된 파일명의 ZIP 파일을 특정 경로에 압축 해제합니다.

    Args:
        zip_file_path (str): 압축 해제할 ZIP 파일의 전체 경로 또는 상대 경로.
        extract_path (str): 파일들을 압축 해제할 대상 경로.
    """
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"'{zip_file_path}' 파일이 '{extract_path}'에 성공적으로 압축 해제되었습니다.")
    except FileNotFoundError:
        print(f"오류: '{zip_file_path}' 파일을 찾을 수 없습니다.")
    except zipfile.BadZipFile:
        print(f"오류: '{zip_file_path}'은 유효한 ZIP 파일이 아닙니다.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")

# 압축 파일 경로와 압축 해제할 폴더 경로를 설정합니다.
zip_file = 'open (1).zip'
extract_dir = 'extracted_open_1'

# 압축 해제할 폴더가 없으면 생성합니다.
os.makedirs(extract_dir, exist_ok=True)

# 압축 해제 함수를 호출합니다.
unzip_special_filename(zip_file, extract_dir)
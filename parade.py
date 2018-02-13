import sys
from some_useful_functions import add_postfix_to_path_string


class TooSmallTextError(Exception):
    def __init__(self, msg):
        self.message = msg


def parade(file_name, batch_size, cut_off_first_batch):
    with open(file_name, 'r') as f:
        text = f.read()
    new_text = ''
    length = len(text)
    # print('(parade)length:', length)
    if batch_size > length:
        raise TooSmallTextError('Text in file "%s" is shorter than batch size' % file_name)
    cursors = [i for i in range(batch_size)]
    
    for _ in range(length):
        for batch_elem_idx, cursor in enumerate(cursors):
            new_text += text[cursor]
            cursors[batch_elem_idx] = (cursor + 1) % length
    parade_name = add_postfix_to_path_string(file_name, '_parade')
    if not cut_off_first_batch:
        with open(parade_name, 'w') as f:
            f.write(new_text)
        return parade_name
    else:
        first_batch, new_text = new_text[:batch_size], new_text[batch_size:]
        with open(parade_name, 'w') as f:
            f.write(new_text)
        first_batch_name = add_postfix_to_path_string(parade_name, '_first_batch')
        with open(first_batch_name, 'w') as f:
            f.write(first_batch)
        return [parade_name, first_batch_name]

if __name__ == '__main__':
    file_name = sys.argv[1]
    batch_size = sys.argv[2]
    cut_off_first_batch = sys.argv[3]
    _ = parade(file_name, batch_size, cut_off_first_batch)

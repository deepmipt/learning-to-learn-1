import numpy as np
import inspect
import os
from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.client import device_lib


escape_sequences = ['\\', '\'', '\"', '\a', '\b', '\f', '\n', '\r', '\t', '\v']
skipped_escape_sequences = ['\\', '\'', '\"']
escape_sequences_replacements = {'\\': '\\\\',
                                 '\'': '\\\'',
                                 '\"': '\\\"',
                                 '\a': '\\a',
                                 '\b': '\\b',
                                 '\f': '\\f',
                                 '\n': '\\n',
                                 '\r': '\\r',
                                 '\t': '\\t',
                                 '\v': '\\v'}


class InvalidArgumentError(Exception):
    def __init__(self, msg, value, name, allowed_values):
        super(InvalidArgumentError, self).__init__(msg)
        self._msg = msg
        self._value = value
        self._name = name
        self._allowed_values = allowed_values


class WrongMethodCallError(Exception):
    def __init__(self, msg):
        super(WrongMethodCallError, self).__init__(msg)
        self._msg = msg


def create_vocabulary(text):
    all_characters = list()
    for char in text:
        if char not in all_characters:
            all_characters.append(char)
    return sorted(all_characters, key=lambda dot: ord(dot))


def get_positions_in_vocabulary(vocabulary):
    character_positions_in_vocabulary = dict()
    for idx, char in enumerate(vocabulary):
        character_positions_in_vocabulary[char] = idx
    return character_positions_in_vocabulary


def char2id(char, character_positions_in_vocabulary):
    if char in character_positions_in_vocabulary:
        return character_positions_in_vocabulary[char]
    else:
        print(u'Unexpected character: %s\nUnexpected character number: %s\n' %
              (repr(char), str([ord(c) for c in char])))
        raise ValueError
        return None


def id2char(dictid, vocabulary):
    voc_size = len(vocabulary)
    if (dictid >= 0) and (dictid < voc_size):
        return vocabulary[dictid]
    else:
        print(u"unexpected id")
        return u'\0'


def filter_text(text, allowed_letters):
    new_text = ""
    for char in text:
        if char in allowed_letters:
            new_text += char
    return new_text


def char2vec(char, character_positions_in_vocabulary):
    voc_size = len(character_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size), dtype=np.float)
    vec[0, char2id(char, character_positions_in_vocabulary)] = 1.0
    return vec


def pred2vec(pred):
    shape = pred.shape
    vecs = np.zeros(shape, dtype=np.float32)
    ids = np.argmax(pred, 1)
    for char_idx, char_id in enumerate(np.nditer(ids)):
        vecs[char_idx, char_id] = 1.
    return vecs


def vec2char(pred, vocabulary):
    char_list = list()
    ids = np.argmax(pred, 1)
    for id in np.nditer(ids):
        char_list.append(id2char(id, vocabulary))
    if len(char_list) > 1:
        return char_list[0]
    else:
        return char_list


def char_2_base_vec(character_positions_in_vocabulary,
                    char):
    voc_size = len(character_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size), dtype=np.float32)
    vec[0, char2id(char, character_positions_in_vocabulary)] = 1.0
    return vec


def create_and_save_vocabulary(input_file_name,
                               vocabulary_file_name):
    input_f = open(input_file_name, 'r', encoding='utf-8')
    text = input_f.read()
    output_f = open(vocabulary_file_name, 'w', encoding='utf-8')
    vocabulary = create_vocabulary(text)
    vocabulary_string = ''.join(vocabulary)
    output_f.write(vocabulary_string)
    input_f.close()
    output_f.close()


def load_vocabulary_from_file(vocabulary_file_name):
    input_f = open(vocabulary_file_name, 'r', encoding='utf-8')
    vocabulary_string = input_f.read()
    return list(vocabulary_string)


def check_not_one_byte(text):
    not_one_byte_counter = 0
    max_character_order_index = 0
    min_character_order_index = 2 ** 16
    present_characters = [0] * 256
    number_of_characters = 0
    for char in text:
        if ord(char) > 255:
            not_one_byte_counter += 1
        if len(present_characters) <= ord(char):
            present_characters.extend([0] * (ord(char) - len(present_characters) + 1))
            present_characters[ord(char)] = 1
            number_of_characters += 1
        elif present_characters[ord(char)] == 0:
            present_characters[ord(char)] = 1
            number_of_characters += 1
        if ord(char) > max_character_order_index:
            max_character_order_index = ord(char)
        if ord(char) < min_character_order_index:
            min_character_order_index = ord(char)
    return not_one_byte_counter, min_character_order_index, max_character_order_index, number_of_characters, present_characters


def construct(obj):
    """Used for preventing of not expected changing of class attributes"""
    if isinstance(obj, OrderedDict):
        new_obj = OrderedDict()
        for key, value in obj.items():
            new_obj[key] = construct(value)
    elif not isinstance(obj, OrderedDict) and isinstance(obj, dict):
        new_obj = dict()
        for key, value in obj.items():
            new_obj[key] = construct(value)
    elif isinstance(obj, list):
        new_obj = list()
        for value in obj:
            new_obj.append(construct(value))
    elif isinstance(obj, tuple):
        base = list()
        for value in obj:
            base.append(construct(value))
        new_obj = tuple(base)
    elif isinstance(obj, str):
        new_obj = str(obj)
    elif isinstance(obj, (int, float, complex, type(None))) or inspect.isclass(obj):
        new_obj = obj
    else:
        raise TypeError("Object of unsupported type was passed to construct function: %s" % type(obj))
    return new_obj


def maybe_download(filename, expected_bytes):
    # Download a file if not present, and make sure it's the right size.
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    if not os.path.exists('enwik8'):
        f = zipfile.ZipFile(filename)
        for name in f.namelist():
            full_text = tf.compat.as_str(f.read(name))
        f.close()
        """f = open('enwik8', 'w')
        f.write(text.encode('utf8'))
        f.close()"""
    else:
        f = open('enwik8', 'r')
        full_text = f.read().decode('utf8')
        f.close()
    return full_text

    f = codecs.open('enwik8', encoding='utf-8')
    text = f.read()
    f.close()
    return text


def flatten(nested):
    if not isinstance(nested, (tuple, list)):
        return [nested]
    output = list()
    for inner_object in nested:
        flattened = flatten(inner_object)
        output.extend(flattened)
    return output


def loop_through_indices(filename, start_index):
    path, name = split_to_path_and_name(filename)
    if '.' in name:
        inter_list = name.split('.')
        extension = inter_list[-1]
        base = '.'.join(inter_list[:-1])
        base += '#%s'
        name = '.'.join([base, extension])

    else:
        name += '#%s'
    if path != '':
        base_path = '/'.join([path, name])
    else:
        base_path = name
    index = start_index
    while os.path.exists(base_path % index):
        index += 1
    return base_path % index


def add_index_to_filename_if_needed(filename, index=None):
    if index is not None:
        return loop_through_indices(filename, index)
    if os.path.exists(filename):
        return loop_through_indices(filename, 1)
    return filename


def split_to_path_and_name(path):
    parts = path.split('/')
    name = parts[-1]
    path = '/'.join(parts[:-1])
    return path, name


def create_path(path, file_name_is_in_path=False):
    if file_name_is_in_path:
        folder_list = path.split('/')[:-1]
    else:
        folder_list = path.split('/')
    if len(folder_list) > 0:
        if folder_list[0] == '':
            current_folder = '/'
        else:
            current_folder = folder_list[0]
        for idx, folder in enumerate(folder_list):
            if idx > 0:
                current_folder += ('/' + folder)
            if not os.path.exists(current_folder):
                os.mkdir(current_folder)


def add_postfix_to_path_string(path, postfix):
    # print('(add_postfix_to_path_string)path:', path)
    # print('(add_postfix_to_path_string)postfix:', postfix)
    dirs_and_file = path.split('/')
    dirs = dirs_and_file[:-1]
    file_name = dirs_and_file[-1]
    # print('(add_postfix_to_path_string)dirs:', dirs)
    # print('(add_postfix_to_path_string)file_name:', file_name)
    if '.' in file_name:
        name_and_ext = file_name.split('.')
        name = '.'.join(name_and_ext[:-1])
        ext = name_and_ext[-1]
        # print('(add_postfix_to_path_string)ext:', ext)
        new_name = name + postfix + '.' + ext
        # print('(add_postfix_to_path_string)new_name:', new_name)
    else:
        new_name = file_name + postfix
    new_path = '/'.join(dirs + [new_name])
    return new_path


def compute_perplexity(probabilities):
    probabilities[probabilities < 1e-10] = 1e-10
    log_probs = np.log2(probabilities)
    entropy_by_character = np.sum(- probabilities * log_probs, axis=1)
    return np.mean(np.exp2(entropy_by_character))


def compute_loss(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    log_predictions = np.log(predictions)
    bpc_by_character = np.sum(- labels * log_predictions, axis=1)
    return np.mean(bpc_by_character)


def compute_bpc(predictions, labels):
    return compute_loss(predictions, labels) / np.log(2)


def compute_accuracy(predictions, labels):
    num_characters = predictions.shape[0]
    num_correct = 0
    for i in range(num_characters):
        if labels[i, np.argmax(predictions, axis=1)[i]] == 1:
            num_correct += 1
    return float(num_correct) / num_characters


def match_two_dicts(small_dict, big_dict):
    """Compares keys of small_dict to keys of big_dict and if in small_dict there is a key missing in big_dict throws
    an error"""
    big_dict_keys = big_dict.keys()
    for key in small_dict.keys():
        if key not in big_dict_keys:
            raise KeyError("Wrong argument name '%s'" % key)
    return True


def split_dictionary(dict_to_split, bases):
    """Function takes dictionary dict_to_split and splits it into several dictionaries according to keys of dicts
    from bases"""
    dicts = list()
    for base in bases:
        if isinstance(base, dict):
            base_keys = base.keys()
        else:
            base_keys = base
        new_dict = dict()
        for key, value in dict_to_split.items():
            if key in base_keys:
                new_dict[key] = value
        dicts.append(new_dict)
    return dicts


def link_into_dictionary(old_dictionary, old_keys, new_key):
    """Used in _parse_train_method_arguments to united several kwargs into one dictionary
    Args:
        old_dictionary: a dictionary which entries are to be united
        old_keys: list of keys to be united
        new_key: the key of new entry  containing linked dictionary"""
    linked = dict()
    for old_key in old_keys:
        if old_key in linked:
            linked[old_key] = old_dictionary[old_key]
            del old_dictionary[old_key]
    old_dictionary[new_key] = linked
    return old_dictionary


def paste_into_nested_structure(structure, searched_key, value_to_paste):
    #print('***********************')
    if isinstance(structure, dict):
        for key, value, in structure.items():
            #print('key:', key)
            if key == searched_key:
                structure[key] = construct(value_to_paste)
            else:
                if isinstance(value, (dict, list, tuple)):
                    paste_into_nested_structure(value, searched_key, value_to_paste)
    elif isinstance(structure, (list, tuple)):
        for elem in structure:
            paste_into_nested_structure(elem, searched_key, value_to_paste)


def check_if_key_in_nested_dict(dictionary, keys):
    new_key_list = keys[1:]
    if keys[0] not in dictionary:
        return False
    if len(new_key_list) == 0:
        return True
    value = dictionary[keys[0]]
    if not isinstance(value, dict):
        return False
    return check_if_key_in_nested_dict(value, new_key_list)


def search_in_nested_dictionary(dictionary, searched_key):
    for key, value in dictionary.items():
        if key == searched_key:
            return value
        else:
            if isinstance(value, dict):
                returned_value = search_in_nested_dictionary(value, searched_key)
                if returned_value is not None:
                    return returned_value
    return None


def add_missing_to_list(extended_list, added_list):
    for elem in added_list:
        if elem not in extended_list:
            extended_list.append(elem)
    return extended_list


def print_and_log(*inputs, log=True, _print=True, fn=None):
    if _print:
        print(*inputs)
    if log:
        for inp in inputs:
            with open(fn, 'a') as fd:
                fd.write(str(inp))
        with open(fn, 'a') as fd:
            fd.write('\n')


def apply_temperature(array, axis, temperature):
    array = np.power(array, 1/temperature)
    norm = np.sum(array, axis=axis, keepdims=True)
    return array / norm


def compute_num_of_repeats(start_axis, last_axis_plus_1, removed_axis, shape):
    num_of_repeats = 1
    for ax_num in range(start_axis, last_axis_plus_1):
        if ax_num != removed_axis:
            num_of_repeats *= shape[ax_num]
    return num_of_repeats


def construct_indices(constructed_axis, removed_axis, shape):
    num_one_value_repeats = compute_num_of_repeats(constructed_axis+1, len(shape), removed_axis, shape)
    num_pattern_repeats = compute_num_of_repeats(0, constructed_axis, removed_axis, shape)
    pattern = list()
    for i in range(shape[constructed_axis]):
        pattern.extend([i] * num_one_value_repeats)
    pattern = pattern * num_pattern_repeats
    return np.array(pattern)


def sample(array, axis):
    shape = array.shape
    if axis == -1:
        axis = len(shape) - 1
    c = array.cumsum(axis=axis)
    r_shape = list(shape)
    r_shape[axis] = 1
    r = np.random.rand(*r_shape)
    mask = (r < c).argmax(axis)
    axis_indices = np.reshape(mask, (-1))
    all_indices = list()
    for ax_num in range(len(shape)):
        if ax_num != axis:
            all_indices.append(construct_indices(ax_num, axis, shape))
        else:
            all_indices.append(axis_indices)
    answer = np.zeros(shape)
    for ax_num in range(len(shape)):
        exec("i%s = all_indices[%s]" % (ax_num, ax_num))
    init_string = ('answer[' + 'i%s' + ', i%s'*(len(shape)-1) + '] = 1.') % tuple([i for i in range(len(shape))])
    exec(init_string)
    return answer


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpu_names = sorted(gpu_names, key=lambda elem: int(elem[5:]))
    return gpu_names


def device_name_scope(device_name):
    return device_name[1:4] + device_name[5:]


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    with tf.name_scope('average_gradients'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


def average_gradients_not_balanced(tower_grads, num_active):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    with tf.name_scope('not_balanced_average'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_sum(grad, 0) / sum(num_active)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


def get_num_gpus_and_bs_on_gpus(batch_size, num_gpus, num_available_gpus):
    batch_sizes_on_gpus = list()
    if num_available_gpus < num_gpus:
        print("Can't build model on %s gpus: only %s gpus are available.\n\tFalling to %s gpus" %
              (num_gpus, num_available_gpus, num_available_gpus))
        num_gpus = num_available_gpus
    if num_gpus > batch_size:
        num_gpus = batch_size
        print("Can't build model on %s gpus: batch size = %s (batch size is to small).\n\tFalling to %s gpus" %
              (num_gpus, batch_size, batch_size))
    else:
        num_gpus = num_gpus
    if num_gpus > 1:
        gpu_batch = batch_size // num_gpus
        num_remaining_elements = batch_size
        for _ in range(num_gpus - 1):
            batch_sizes_on_gpus.append(gpu_batch)
            num_remaining_elements -= gpu_batch
        batch_sizes_on_gpus.append(num_remaining_elements)
    else:
        batch_sizes_on_gpus = [batch_size]
    return num_gpus, batch_sizes_on_gpus


def nested2string(nested, indent=0):
    string = list()
    indent = indent
    recur_nested_2_string(nested, string, indent)
    string = ''.join(string)
    return string


def recur_nested_2_string(nested, string, indent):
    ind = ' ' * indent
    if not isinstance(nested, (list, tuple, dict)):
        if isinstance(nested, str):
            if len(nested) > 50:
                string.append('VERY_LONG_STRING')
            else:
                string.append('\"' + add_escape_characters(nested) + '\"')
        else:
            string.append(str(nested))
    elif isinstance(nested, dict):
        string.append('{')
        if len(nested) > 0:
            string.append('\n')
        for idx, (key, value) in enumerate(nested.items()):
            if isinstance(key, str):
                string.append(ind + ' ' *4 + '\"' + key + '\"' + ': ')
            else:
                string.append(ind + ' ' *4 + str(key) + ': ')
            recur_nested_2_string(value, string, indent + 4)
            if idx < len(nested) - 1:
                string.append(',\n')
        string.append('}')
    elif isinstance(nested, list):
        string.append('[')
        if len(nested) > 0:
            string.append('\n')
        for idx, value in enumerate(nested):
            string.append(ind + ' ' *4)
            recur_nested_2_string(value, string, indent + 4)
            if idx < len(nested) - 1:
                string.append(',\n')
        string.append(']')

    elif isinstance(nested, tuple):
        string.append('(')
        if len(nested) > 0:
            string.append('\n')
        else:
            string.append(',')
        for idx, value in enumerate(nested):
            string.append(ind + ' ' *4)
            recur_nested_2_string(value, string, indent + 4)
            if idx < len(nested) - 1:
                string.append(',\n')
        string.append(')')


def add_escape_characters(string):
    new_string = ''
    for char in string:
        if char in escape_sequences and char not in skipped_escape_sequences:
            new_string += escape_sequences_replacements[char]
        else:
            new_string += char
    return new_string


def all_entries_in_nested_structure(nested):
    all_entries = list()
    recur_entries(nested, all_entries)
    return all_entries


def recur_entries(nested, all_entries):
    if isinstance(nested, dict):
        all_entries.extend(list(nested.keys()))
        for value in nested.values():
            recur_entries(value, all_entries)
    elif isinstance(nested, (list, tuple)):
        for value in nested:
            recur_entries(value, all_entries)


def unite_dicts(list_of_dicts):
    new_dict = OrderedDict()
    for d in list_of_dicts:
        new_dict.update(d)
    return new_dict


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

import time
from data_generator import *
from network import *
from torch.autograd import Variable
from graphviz import Digraph
import random
from functools import reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging
logging.basicConfig(level=logging.ERROR)


def showPlot(*curves):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    for curve in curves:
        plt.plot(curve)
    plt.savefig('loss curve.png')


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(_var):
        if _var not in seen:
            if torch.is_tensor(_var):
                dot.node(str(id(_var)), size_to_str(_var.size()), fillcolor='orange')
            elif hasattr(_var, 'variable'):
                u = _var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(_var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(_var)), str(type(_var).__name__))
            seen.add(_var)
            if hasattr(_var, 'next_functions'):
                for u in _var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(_var)))
                        add_nodes(u[0])
            if hasattr(_var, 'saved_tensors'):
                for t in _var.saved_tensors:
                    dot.edge(str(id(t)), str(id(_var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


def initial_hiddens(batch_size, hidden_sizes, device):
    n_layers = len(hidden_sizes)
    h = []
    c = []
    m = []
    if device is None:
        for i in range(n_layers):
            h.append(torch.zeros([batch_size, *hidden_sizes[i]]))
            c.append(torch.zeros([batch_size, *hidden_sizes[i]]))
            m.append(torch.zeros([batch_size, *hidden_sizes[i]]))
        z = torch.zeros(batch_size, *hidden_sizes[0])
        return h, c, m, z
    else:
        for i in range(n_layers):
            h.append(torch.zeros([batch_size, *hidden_sizes[i]], device=device))
            c.append(torch.zeros([batch_size, *hidden_sizes[i]], device=device))
            m.append(torch.zeros([batch_size, *hidden_sizes[i]], device=device))
        z = torch.zeros(batch_size, *hidden_sizes[0], device=device)
        return h, c, m, z


def activation_statistics(states):
    h, c, m, z = states
    h_avg = []
    c_avg = []
    m_avg = []
    z_avg = None
    n_layers = len(h)
    for s, s_avg in zip((h, c, m), (h_avg, c_avg, m_avg)):
        for i in range(n_layers):
            s_avg.append(round(torch.sum(s[i]).item()/reduce(lambda x, y: x*y, list(s[i].size()))), 4)
    z_avg = round(torch.sum(z).item()/reduce(lambda x, y: x*y, list(z.size()), 4))
    return h_avg, c_avg, m_avg, z_avg


def grad_statistics(model):
    total_grad_value = 0.0
    num_variables = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_value += torch.sum(torch.abs(param.grad)).item()
            num_variables += reduce(lambda x, y: x*y, list(param.grad.size()))

    return total_grad_value / num_variables


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, hidden_sizes, device, thresh=10, teacher_forcing_ratio=0.5):
    encoder_optimizer.zero_grad()
    if decoder is not None:
        decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden = initial_hiddens(input_variable.size()[1], hidden_sizes, device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)

        logging.debug("input step{}".format(ei))
        logging.debug("h {}\n c {}\n m {}\n z {}".format(*activation_statistics(encoder_hidden)))

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    loss = 0.0
    if decoder is None:
        for di in range(target_length):
            loss += criterion(encoder_output, target_variable[di])
            if use_teacher_forcing:
                encoder_input = target_variable[di]
            else:
                encoder_input = encoder_output
            encoder_output, encoder_hidden = encoder(encoder_input, encoder_hidden)

            logging.debug("output step{}".format(di))
            logging.debug("h {}\n c {}\n m {}\n z {}".format(*activation_statistics(encoder_hidden)))

        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), thresh)

        logging.debug("encoder grad statistics {}".format(grad_statistics(encoder)))

        encoder_optimizer.step()
    else:
        decoder_input = torch.zeros_like(input_variable[0])

        decoder_hidden = encoder_hidden
        decoder_context = encoder_output  # the last output

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_context, decoder_hidden)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]

                logging.debug("output step{}".format(di))
                logging.debug("h {}\n c {}\n m {}\n z {}".format(*activation_statistics(decoder_hidden)))

        else:
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_context, decoder_hidden)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = decoder_output

                logging.debug("output step{}".format(di))
                logging.debug("h {}\n c {}\n m {}\n z {}".format(*activation_statistics(decoder_hidden)))

        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), thresh)
        nn.utils.clip_grad_norm_(decoder.parameters(), thresh)
        encoder_optimizer.step()
        decoder_optimizer.step()

        logging.debug("encoder grad statistics {}".format(grad_statistics(encoder)))
        logging.debug("decoder grad statistics {}".format(grad_statistics(decoder)))

    return loss.item()


def trainIters(encoder, decoder, train_criterion, train_data_generator, hidden_sizes, n_iters, device=None, print_every=20, plot_every=100, learning_rate=0.01):
    start_time = time.time()
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    if decoder is not None:
        decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    else:
        decoder_optimizer = None

    print_loss_total = 0.0
    plot_loss_total = 0.0
    plot_losses = []

    for it in range(1, n_iters+1):
        train_pair = train_data_generator.GetBatch()
        input_variable = train_pair[0]
        target_variable = train_pair[1]
        if device is not None:
            input_variable = input_variable.to(device)
            target_variable = input_variable.to(device)

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, train_criterion, hidden_sizes, device)

        print_loss_total += loss
        plot_loss_total += loss

        if it % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0.0
            print('%s (%d %d%%) %.4f' % (timeSince(start_time, it / n_iters),
                                         it, it / n_iters * 100, print_loss_avg))

        if it % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0.0

    showPlot(plot_losses)


def evaluate(encoder, decoder, criterion, eval_data_generator, hidden_sizes, device):
    eval_pair = eval_data_generator.GetBatch()
    input_variable = eval_pair[0]
    target_variable = eval_pair[1]

    batch_size = input_variable.size()[1]
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    encoder_hidden = initial_hiddens(batch_size, hidden_sizes, device)

    if device is not None:
        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)

    loss = 0.0
    if decoder is None:
        for di in range(target_length):
            loss += criterion(encoder_output, target_variable[di])
            encoder_output, encoder_hidden = encoder(encoder_output, encoder_hidden)
        print('backward called')
        loss.backward()
    else:
        decoder_input = torch.zeros_like(input_variable[0])

        decoder_hidden = encoder_hidden
        decoder_context = encoder_output  # the last output as context

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_context, decoder_hidden)

            decoder_input = decoder_output

            loss += criterion(decoder_output, target_variable)

    return loss.item() / batch_size


def show_generation(encoder, input_variable, hidden_sizes, device):
    plt.figure(figsize=(20, 1))
    encoder_hidden = initial_hiddens(1, hidden_sizes, device)
    if device is not None:
        input_variable = input_variable.to(device)

    for ei in range(10):
        plt.subplot(1, 20, 1 + ei)
        plt.imshow(input_variable[ei, 0, 0].detach())
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
    for di in range(10):
        plt.subplot(1, 20, 11 + di)
        plt.imshow(encoder_output[0, 0].detach())
        encoder_output, encoder_hidden = encoder(encoder_output, encoder_hidden)
    plt.savefig("generated_images.png")


def main():
    para_dir = "./parameters/model.pkl"

    train_batch_size = 8
    eval_batch_size = 16
    train_generator = BouncingMNISTDataHandler(train_batch_size, 2)
    eval_generator = BouncingMNISTDataHandler(eval_batch_size, 3)

    input_size = (1, 64, 64)
    hidden_sizes = [(128, 64, 64), (32, 64, 64), (32, 64, 64), (32, 64, 64)]
    kernel_HWs = [(5, 5)]*4

    encoder = Encoder(input_size, hidden_sizes, kernel_HWs)
    encoder = torch.nn.DataParallel(encoder)
    try:
        encoder.load_state_dict(torch.load(para_dir))
    except FileNotFoundError:
        pass

    device = None
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:0")
        encoder.to(device)

    L1_loss = nn.L1Loss()
    L2_loss = nn.MSELoss()
    L2_loss_ = nn.MSELoss()
    train_criterion = lambda inputs, targets: L1_loss(inputs, targets) + L2_loss(inputs, targets)
    trainIters(encoder, None, train_criterion, train_generator, hidden_sizes, 2000, device)
    torch.save(encoder.state_dict(), para_dir)
    print('eval loss', evaluate(encoder, None, L2_loss_, eval_generator, hidden_sizes, device))

    # input_var = BouncingMNISTDataHandler(1, 2).GetBatch()[0]
    # show_generation(encoder, input_var, hidden_sizes, device)

if __name__ == '__main__':
    main()

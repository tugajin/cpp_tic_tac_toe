#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include "../ai/common.hpp"
#include "../ai/game.hpp"
#include "../ai/movelegal.hpp"
#include "../ai/hash.hpp"
#include "../ai/nn.hpp"

namespace py = pybind11;

TeeStream Tee;


PYBIND11_MODULE(gamelibs, m) {
    
    m.doc() = "gamelibs made by pybind11";

    py::enum_<Color>(m, "Color", py::arithmetic())
        .value("BLACK", Color::BLACK)
        .value("WHITE", Color::WHITE);

    py::enum_<Move>(m, "Move", py::arithmetic())
        .value("MOVE_NONE", Move::MOVE_NONE);

    m.def("val_to_move",[](int m){
        return Move(m);
    });
    
    m.def("move_to_val",[](Move x){
        return static_cast<int>(x);
    });

    m.def("from_hash",&hash::from_hash);
    m.def("hirate", &hash::hirate);

    m.def("hash_key", &hash::hash_key);
    m.def("legal_moves", &gen::legal_moves);
    m.def("feature", &nn::feature);

    py::class_<movelist::MoveList>(m, "MoveList")
        .def(py::init<>())
        .def("init", &movelist::MoveList::init)
        .def("add", &movelist::MoveList::add)
        .def("begin", &movelist::MoveList::begin)
        .def("end", &movelist::MoveList::end)
        .def("len", &movelist::MoveList::len)
        .def("__str__", &movelist::MoveList::str)
        .def("__getitem__", &movelist::MoveList::operator[]);

    py::class_<game::Position>(m, "Position")
        .def(py::init<>())
        .def("turn", &game::Position::turn)
        .def("next",&game::Position::next)
        .def("__str__",&game::Position::str)
        .def("is_win",&game::Position::is_win)
        .def("is_draw",&game::Position::is_draw)
        .def("is_lose",&game::Position::is_lose)
        .def("is_done",&game::Position::is_done);

}
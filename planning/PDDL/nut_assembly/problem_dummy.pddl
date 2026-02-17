(define (problem nut_assembly_problem)
  (:domain nut_assembly)
  (:objects 
    RoundNut - round_nut
    SquareNut - square_nut
    RoundPeg - round_peg
    SquarePeg - square_peg
    table1 - table ;change the name of table to avoid conflict with the table type when parsing
  )
  (:init 
    (free-gripper)
    (clear RoundNut)
    (clear SquareNut)
    (clear RoundPeg)
    (clear SquarePeg)
    (on SquareNut table1)
    (on RoundNut table1)
    (matches RoundNut RoundPeg)
    (matches SquareNut SquarePeg)
  )
  (:goal 
    (and (on RoundNut RoundPeg)
         (on SquareNut SquarePeg))
  )
)

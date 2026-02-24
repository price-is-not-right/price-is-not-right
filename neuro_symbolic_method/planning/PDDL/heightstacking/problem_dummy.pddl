(define (problem heightstacking)
  (:domain heightstacking)
  (:objects 
    cube0 cube1 cube2 cube3 cube4 cube5 - disk
    platform table - peg
  )
  (:init 
(smaller cube2 platform )
(smaller cube1 platform )
(smaller cube1 cube2 )
(smaller cube0 platform )
(smaller cube0 cube2 )
(smaller cube0 cube1 )
(clear cube2 )
(clear cube1 )
(clear cube0 )
    (free-gripper)
    (on cube0 table)
    (on cube1 table)
    (on cube2 table)
    (on cube3 table)
    (on cube4 table)
    (on cube5 table)
    (clear platform)
  )
  (:goal 
    (and
      (on cube0 cube1)
      (on cube1 cube2)
      (on cube2 platform)
    )
  )
)

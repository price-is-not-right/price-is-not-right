(define (problem patternreplication)
  (:domain patternreplication)
  (:objects 
    cube0 cube1 cube2 - disk
    target_platform table - peg
  )
  (:init 
(clear target_platform )
(clear cube2 )
(clear cube1 )
(clear cube0 )
    (free-gripper)
    (on cube0 table)
    (on cube1 table)
    (on cube2 table)
  )
  (:goal 
    (and
      (on cube1 cube2)
      (on cube2 cube0)
      (on cube0 target_platform)
    )
  )
)

(define (problem assemblyline)
  (:domain assemblyline)
  (:objects 
    cube0 cube1 cube2 cube3 cube4 cube5 - disk
    bin0 bin1 bin2 bin3 bin4 bin5 table - peg
  )
  (:init 
    (free-gripper)    
    (on cube0 table)
    (on cube1 table)
    (on cube2 table)
    (on cube3 table)
    (on cube4 table)
    (on cube5 table)
  )
  (:goal 

  )
)
